import json
import logging
import os
import random
import sys
from pathlib import Path

# --- ML & Data Libraries ---
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
import wandb
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

# --- Configuration & Setup ---
import hydra
from omegaconf import DictConfig, OmegaConf


# Setup del root del progetto nativo
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT))

from bev.models.bevqa import get_model
from bev.data.bevqa_dataset import BEVQADataset
from bev.training.train import train_epoch, val_epoch
from bev.utils.text_utils import decode_question

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    run_name = f"run_{cfg.model.type}_{cfg.run_name}"

    wandb.init(
        project="BEV-VQA",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")
    log.info(f"Working directory: {os.getcwd()}")

    # Path
    feat_dir = Path(cfg.paths.bev_features_dir)
    dict_dir = Path(cfg.paths.dict_dir)
    glove_path = Path(cfg.paths.glove_path)
    assert feat_dir.exists(), f"[ERROR]: Features folder not found -> {feat_dir}"
    assert dict_dir.exists(), f"[ERROR]:  Dict folder not found -> {dict_dir}"
    assert glove_path.exists(), f"[ERROR]: Glove file not found -> {glove_path}"

    # Dataset
    train_dataset = BEVQADataset(
        bev_dir=feat_dir / "train",
        json_path=dict_dir / "NuScenes_train_questions.json",
        glove=glove_path,
        answer2idx=None # Il train crea il mapping
    )
    val_dataset = BEVQADataset(
        bev_dir=feat_dir / "val",
        json_path=dict_dir / "NuScenes_val_questions.json",
        glove=glove_path,
        answer2idx=train_dataset.answer2idx # Il val usa lo stesso mapping del train
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    # Modello
    text_cfg = cfg.model.get("text", None)
    if text_cfg is not None:
        text_cfg = OmegaConf.to_container(text_cfg, resolve=True)
    model = get_model(cfg.model.type, text_config=text_cfg).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=cfg.training.scheduler_factor, 
        patience=cfg.training.scheduler_patience
    )

    scaler = torch.amp.GradScaler('cuda') if cfg.training.use_amp and device == "cuda" else None

    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    early_stop_patience = cfg.training.early_stopping_patience

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # --- LOOP DI TRAINING ---
    for epoch in range(cfg.training.num_epochs):
        log.info(f"Epoch {epoch+1}/{cfg.training.num_epochs}")
        
        tr_loss, tr_acc, tr_time = train_epoch(
            model, train_dataloader, optimizer, criterion, device, 
            scaler=scaler, grad_clip=cfg.training.gradient_clip
        )
        val_loss, val_acc, val_time = val_epoch(
            model, val_dataloader, criterion, device, 
            use_amp=(scaler is not None)
        )
        
        scheduler.step(val_loss)
        
        log.info(f"Train Loss: {tr_loss:.4f} - Acc: {tr_acc:.4f} - Time: {tr_time:.2f}s")
        log.info(f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f} - Time: {val_time:.2f}s")

        # Log REAL-TIME su WandB
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": tr_loss,
            "train/acc": tr_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })
        
        # Salvataggio miglior modello
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = ckpt_dir / f"best_model_{cfg.model.type}.pth"
            torch.save(model.state_dict(), save_path)
            log.info(f"  → Saved best model to {save_path.absolute()}")
            wandb.run.summary["best_val_acc"] = val_acc

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                log.info(f"Early stopping triggered after {epoch+1} epochs!")
                break
    
    # --- ANALISI FINALE: SOLO LOCALE ---
    log.info("Generating Final Analysis (Local files only)...")
    try:
        # Carica il miglior modello
        best_model_path = ckpt_dir / f"best_model_{cfg.model.type}.pth"
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for feat, quest, ans in val_dataloader:
                out = model(feat.to(device), quest.to(device))
                all_preds.extend(out.argmax(1).cpu().numpy().tolist())
                all_targets.extend(ans.numpy().tolist())
        
        # Mapping classi completo (usando il training JSON per sicurezza)
        with open(dict_dir / "NuScenes_train_questions.json", "r") as f:
            train_json_data = json.load(f)
        _, idx2answer = decode_question(train_json_data)
        class_names = [idx2answer.get(i, f"ID_{i}") for i in range(len(idx2answer))]
        
        # 1. Confusion Matrix Locale
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {cfg.model.type}')
        plt.tight_layout()
        plt.savefig("confusion_matrix_local.png") # Salvataggio locale
        plt.close()
        
        # 2. Visual Samples Locali
        for i in range(5):
            idx = random.randint(0, len(val_dataset) - 1)
            sample = val_dataset.samples[idx]
            token = sample["token"]
            question_str = sample["question"]
            gt_answer_str = sample["answer"]
            
            features, question_enc, _ = val_dataset[idx]
            
            with torch.no_grad():
                out = model(features.unsqueeze(0).to(device), question_enc.unsqueeze(0).to(device))
                pred_idx = out.argmax(1).item()
                pred_answer = idx2answer.get(pred_idx, "UNKNOWN")
            
            bev_img = features.mean(dim=0).cpu().numpy()
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(bev_img, cmap="viridis")
            ax.set_title(f"Sample {i+1} | Token: {token}")
            
            info_text = f"Q: {question_str}\nGT: {gt_answer_str}\nPRED: {pred_answer}"
            plt.figtext(0.5, 0.01, info_text, wrap=True, horizontalalignment='center', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            
            plt.subplots_adjust(bottom=0.15)
            plt.savefig(f"sample_result_{i+1}_local.png") # Salvataggio locale
            plt.close()
            
        # 3. Metriche Finali JSON Locale
        final_results = {
            "best_val_acc": float(best_val_acc),
            "final_val_loss": float(val_loss),
            "model_type": cfg.model.type
        }
        with open("final_metrics.json", "w") as f:
            json.dump(final_results, f, indent=4)
            
        log.info(f"Final Analysis complete. Local files saved in current Hydra directory: {os.getcwd()}")

    except Exception as e:
        log.error(f"Error during final analysis: {e}")

    # Chiudi WandB
    wandb.finish()

if __name__ == "__main__":
    main()
