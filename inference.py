"""
Inference/test finale: carica un checkpoint, valuta su un intero split (default: val)
e salva tutte le predizioni + accuracy su file, per analisi successive.

Uso:
    uv run test_inference.py
    uv run test_inference.py --ckpt checkpoints/best_vlm.pth --split val --out results.json
"""
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm

from src.bev.data.dataset import BEVQADataset, build_tokenizer, collate_eval, normalize_answer
from src.bev.models.model import BEVVLM, BEVVLMConfig


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    bev_token_id = ckpt["bev_token_id"]

    tok, _ = build_tokenizer(cfg.llm.name, n_bev_tokens=cfg.model.num_queries)

    model_cfg = BEVVLMConfig(
        llm_name=cfg.llm.name, bev_token_id=bev_token_id, vocab_size=len(tok),
        vision_hidden_size=cfg.model.vision_hidden_size,
        vision_num_layers=cfg.model.vision_num_layers,
        vision_num_heads=cfg.model.vision_num_heads,
        num_queries=cfg.model.num_queries,
        lora_r=cfg.model.lora.r, lora_alpha=cfg.model.lora.alpha,
        lora_dropout=cfg.model.lora.dropout,
    )
    model = BEVVLM(model_cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
    print(f"Pesi caricati | mancanti: {len(missing)} | inattesi: {len(unexpected)}")
    model.eval()
    return model, tok, cfg


@torch.no_grad()
def run_inference(model, loader, tok, device, max_new_tokens):
    results = []
    for batch in tqdm(loader, desc="Inference"):
        bev = batch["bev"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            gen = model.generate(
                bev, input_ids, attn,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        texts = tok.batch_decode(gen, skip_special_tokens=True)

        for q, gt, gen_txt, tmpl, tok_id in zip(batch["questions"], batch["answers"], texts,
                                          batch["template_types"], batch["sample_tokens"]):
            gen_txt = gen_txt.strip()
            results.append({
                "question": q, "gt": gt, "gen": gen_txt, "template_type": tmpl,
                "sample_token": tok_id,
                "correct": normalize_answer(gen_txt) == normalize_answer(gt),
            })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/best_vlm.pth")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--max_samples", type=int, default=None, help="limita il numero di esempi (debug)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, cfg = load_model(args.ckpt, device)

    feat_dir = Path(cfg.paths.bev_features_dir)
    dict_dir = Path(cfg.paths.dict_dir)
    ds = BEVQADataset(
        feat_dir / args.split, dict_dir / f"NuScenes_{args.split}_questions.json",
        tok, n_bev_tokens=cfg.model.num_queries, max_len=cfg.training.max_len,
        cache_dir=Path(cfg.paths.dataset_dir) / "cache",
        fraction=1.0,  # inferenza finale: sempre sul set completo, non sottocampionato
    )
    print(f"{args.split} dataset: {len(ds)} esempi")

    if args.max_samples:
        ds.samples = ds.samples[:args.max_samples]

    loader = DataLoader(
        ds, batch_size=cfg.training.eval_batch_size, shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=partial(collate_eval, pad_id=tok.pad_token_id),
    )

    results = run_inference(model, loader, tok, device, cfg.training.max_new_tokens)

    acc = sum(r["correct"] for r in results) / len(results)
    print(f"\nAccuracy su {len(results)} esempi: {acc:.4f}")

    with open(args.out, "w") as f:
        json.dump({"accuracy": acc, "n_samples": len(results), "results": results}, f, indent=2)
    print(f"Risultati salvati in {args.out}")


if __name__ == "__main__":
    main()