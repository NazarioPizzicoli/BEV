import argparse
import logging
import re
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT))

from bev.models.bevqa import get_model
from bev.data.bevqa_dataset import BEVQADataset
from bev.training.train import val_epoch
from bev.utils.text_utils import decode_question

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] - %(message)s")
log = logging.getLogger(__name__)

def parse_log_file(log_path):
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    with open(log_path, 'r') as f:
        for line in f:
            if "Epoch " in line and "/" in line:
                match = re.search(r"Epoch (\d+)/", line)
                if match:
                    epoch = int(match.group(1))
                    if epoch not in epochs:
                        epochs.append(epoch)
            elif "Train Loss:" in line:
                match = re.search(r"Train Loss: ([\d.]+) - Acc: ([\d.]+)", line)
                if match:
                    train_losses.append(float(match.group(1)))
                    train_accs.append(float(match.group(2)))
            elif "Val Loss:" in line:
                match = re.search(r"Val Loss: ([\d.]+) - Acc: ([\d.]+)", line)
                if match:
                    val_losses.append(float(match.group(1)))
                    val_accs.append(float(match.group(2)))

    # Ensure lengths match
    min_len = min(len(epochs), len(train_losses), len(val_losses))
    return epochs[:min_len], train_losses[:min_len], train_accs[:min_len], val_losses[:min_len], val_accs[:min_len]

def plot_metrics(log_path, output_dir):
    epochs, tr_loss, tr_acc, val_loss, val_acc = parse_log_file(log_path)
    
    if not epochs:
        log.warning(f"No metrics found in {log_path}")
        return

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, tr_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Trend')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, tr_acc, label='Train Acc', marker='o')
    plt.plot(epochs, val_acc, label='Val Acc', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Trend')
    plt.legend()
    plt.grid(True)
    
    plot_path = output_dir / "metrics_plot.png"
    plt.savefig(plot_path)
    plt.close()
    log.info(f"Saved metrics plot to {plot_path}")

def generate_sample_visualization(model, val_dataset, json_path, output_dir, device):
    """
    Selects a random sample from the validation set, runs the model,
    and plots the BEV map with the question, predicted answer, and GT answer.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Retrieve idx to answer mapping
    _, idx2answer = decode_question(data)
    
    idx = random.randint(0, len(val_dataset) - 1)
    token, question_str, gt_answer_str = val_dataset.samples[idx]
    
    features, question_enc, gt_idx = val_dataset[idx]
    
    model.eval()
    with torch.no_grad():
        feat_batch = features.unsqueeze(0).to(device)
        quest_batch = question_enc.unsqueeze(0).to(device)
        out = model(feat_batch, quest_batch)
        pred_idx = out.argmax(1).item()
        
    pred_answer_str = idx2answer.get(pred_idx, "UNKNOWN")
    
    # Visualize BEV map: Take the mean across the 128 channels
    bev_img = features.mean(dim=0).cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(bev_img, cmap='viridis')
    plt.colorbar(label='Mean Activation')
    plt.title(f"Sample Token: {token}")
    
    # Add text box with Q&A
    text_str = (
        f"Question: {question_str}\n"
        f"GT Answer: {gt_answer_str}\n"
        f"Predicted: {pred_answer_str}"
    )
    plt.figtext(0.5, 0.02, text_str, wrap=True, horizontalalignment='center', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    
    sample_path = output_dir / "sample_eval.png"
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(sample_path)
    plt.close()
    log.info(f"Saved sample visualization to {sample_path}")


def evaluate_checkpoint(run_dir, checkpoint_path, device="cuda"):
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        log.error(f"Config not found at {config_path}")
        return

    cfg = OmegaConf.load(config_path)
    
    model_type = cfg.model.type
    log.info(f"Model type from config: {model_type}")
    
    # Initialize the model
    text_cfg = cfg.model.get("text", None)
    if text_cfg is not None:
        text_cfg = OmegaConf.to_container(text_cfg, resolve=True)
    model = get_model(model_type, text_config=text_cfg).to(device)
    
    if not checkpoint_path.exists():
        log.error(f"Checkpoint not found at {checkpoint_path}")
        return
        
    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    log.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Reconstruct Dataset
    feat_dir = Path(cfg.paths.bev_features_dir)
    dict_dir = Path(cfg.paths.dict_dir)
    glove_path = Path(cfg.paths.glove_path)

    val_json_path = dict_dir / "NuScenes_val_questions.json"
    log.info(f"Loading validation dataset from {feat_dir / 'val'} ...")
    val_dataset = BEVQADataset(
        bev_dir=feat_dir / "val",
        json_path=val_json_path,
        glove=glove_path
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        num_workers=cfg.training.num_workers
    )

    criterion = nn.CrossEntropyLoss()
    
    log.info("Starting evaluation on validation set...")
    val_loss, val_acc, val_time = val_epoch(
        model, val_dataloader, criterion, device, 
        use_amp=cfg.training.use_amp and device == "cuda"
    )
    
    log.info(f"==================================================")
    log.info(f"FINAL EVALUATION RESULTS:")
    log.info(f"Loss: {val_loss:.4f}")
    log.info(f"Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")
    log.info(f"Time: {val_time:.2f}s")
    log.info(f"==================================================")
    
    # Save results to JSON
    results = {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_time": val_time,
        "model_type": model_type
    }
    results_path = run_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    log.info(f"Saved evaluation results to {results_path}")
    
    # Generate sample visualization
    generate_sample_visualization(model, val_dataset, val_json_path, run_dir, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BEVQA models and plot metrics.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the Hydra run directory (e.g., outputs/mini.yaml/2026-06-15/16-45-52)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint to evaluate (e.g., checkpoints/best_model_cnn.pth)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    checkpoint_path = Path(args.checkpoint)
    
    log_file = run_dir / "main.log"
    if log_file.exists():
        plot_metrics(log_file, run_dir)
    else:
        log.warning(f"Log file {log_file} not found. Skipping plotting.")
        
    evaluate_checkpoint(run_dir, checkpoint_path, args.device)
