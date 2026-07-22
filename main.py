"""
main
Architecture: vision.py -> model.py -> main.py

Usage:
    python main.py paths=mini_veh
    python main.py paths=veh training.batch_size=2 training.grad_accum=8
"""
import logging
import os
from functools import partial
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.bev.data.dataset import BEVQADataset, build_tokenizer, collate_train, collate_eval
from src.bev.models.model import BEVVLM, BEVVLMConfig
from src.bev.training.train import train_epoch, val_loss, evaluate

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    wandb.init(project="BEV-VLM", name=cfg.run_name,
               config=OmegaConf.to_container(cfg, resolve=True))
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device} | cwd: {os.getcwd()}")

    feat_dir = Path(cfg.paths.bev_features_dir)
    dict_dir = Path(cfg.paths.dict_dir)
    assert feat_dir.exists(), f"Features non trovate -> {feat_dir}"
    assert dict_dir.exists(), f"Dict non trovato -> {dict_dir}"

    n_bev = cfg.model.num_queries

    # Tokenizer (+ token <|bev|>)
    tok, bev_token_id = build_tokenizer(cfg.llm.name)

    # Dataset
    train_ds = BEVQADataset(
        feat_dir / "train", dict_dir / "NuScenes_train_questions.json",
        tok, n_bev_tokens=n_bev, max_len=cfg.training.max_len,
        cache_dir=Path(cfg.paths.dataset_dir) / "cache", fraction=cfg.data.train_fraction,
    )
    val_ds = BEVQADataset(
        feat_dir / "val", dict_dir / "NuScenes_val_questions.json",
        tok, n_bev_tokens=n_bev, max_len=cfg.training.max_len,
        cache_dir=Path(cfg.paths.dataset_dir) / "cache", fraction=cfg.data.val_fraction,
    )
    log.info(f"Train: {len(train_ds)} QA | Val: {len(val_ds)} QA | bev_tokens={n_bev}")

    pad = tok.pad_token_id
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, pin_memory=True,
        collate_fn=partial(collate_train, pad_id=pad),
    )
    val_lm_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=partial(collate_train, pad_id=pad),
    )
    val_gen_loader = DataLoader(
        val_ds, batch_size=cfg.training.eval_batch_size, shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=partial(collate_eval, pad_id=pad),
    )

    # Modello
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

    n_train_p = sum(p.numel() for p in model.trainable_parameters())
    log.info(f"Parametri trainabili: {n_train_p/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.num_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_acc = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    patience = cfg.training.early_stopping_patience

    for epoch in range(cfg.training.num_epochs):
        log.info(f"Epoch {epoch+1}/{cfg.training.num_epochs}")
        tr_loss, tr_time = train_epoch(
            model, train_loader, optimizer, scaler, device,
            grad_accum=cfg.training.grad_accum, grad_clip=cfg.training.grad_clip,
        )
        scheduler.step()

        eval_max_batches = cfg.training.eval_max_batches
        v_loss = val_loss(model, val_lm_loader, device, max_batches=eval_max_batches)
        ev = evaluate(
            model, val_gen_loader, tok, device,
            max_new_tokens=cfg.training.max_new_tokens, log_samples=5,
            max_batches=eval_max_batches,
        )

        log.info(f"Train loss {tr_loss:.4f} ({tr_time:.1f}s) | Val loss {v_loss:.4f} | "
                  f"Val EM-acc {ev['acc']:.4f} ({ev['time']:.1f}s)")
        for s in ev["samples"]:
            log.info(f"  Q: {s['question']} | GT: {s['gt']} | GEN: {s['gen']!r}")

        wandb.log({
            "epoch": epoch + 1, "train/loss": tr_loss, "val/loss": v_loss,
            "val/em_acc": ev["acc"], "lr": optimizer.param_groups[0]["lr"],
        })

        # --- best checkpoint su EM-acc ---
        if ev["acc"] > best_acc:
            best_acc = ev["acc"]
            trainable = {k: v.cpu() for k, v in model.state_dict().items()
                         if "vision_model" in k or "lora_" in k}
            torch.save({"state": trainable, "bev_token_id": bev_token_id,
                        "cfg": OmegaConf.to_container(cfg, resolve=True)},
                       ckpt_dir / "best_vlm.pth")
            wandb.run.summary["best_em_acc"] = best_acc
            log.info(f"  -> salvato best (EM-acc {best_acc:.4f})")

        # --- early stopping su val loss ---
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
        else:
            patience_counter += 1
            log.info(f"  -> val loss non migliora ({patience_counter}/{patience})")
            if patience_counter >= patience:
                log.info(f"Early stopping: nessun miglioramento della val loss per {patience} epoche.")
                break

    wandb.finish()


if __name__ == "__main__":
    main()