# BEV-VLM: Bird's-Eye-View Visual Question Answering for Autonomous Driving

A generative Vision-Language Model (image-text-to-text) that answers natural language
questions about driving scenes using Bird's-Eye-View (BEV) feature maps instead of raw
camera images.

## Overview

The project follows the **LLaVA** recipe (visual instruction tuning): a BEV feature map
is encoded into visual tokens via an attention-based Q-Former, projected into the
embedding space of an LLM (Qwen2.5), and spliced into the text prompt. The LLM, adapted
with LoRA, generates free-form natural language answers (not single-word/template answers).

```markdown
## Architecture Overview

```mermaid
graph TD
    A["BEV Features<br/><code>[B, 128, 200, 200]</code>"] -->|Q-Former| B["Query Tokens<br/><code>[B, N, hidden]</code>"]
    B -->|Linear Projection| C["LLM Tokens<br/><code>[B, N, d_llm]</code>"]
    C -->|Spliced into &lt;|bev|&gt;| D["Qwen2.5<br/><i>(Frozen + LoRA)</i>"]
    D --> E["Generated Natural Language Answer"]

The project is inspired by [BeLLA (Mohan et al., 2025)](https://arxiv.org/abs/2512.06096),
from which we adopt selected techniques (in particular BEV-text alignment pretraining),
adapted to our architecture and use case (precomputed BEV features rather than raw
multi-camera images).

## Architecture

| Component | Implementation |
|---|---|
| **Vision encoder** | Q-Former: learnable queries + self/cross-attention over the flattened BEV feature map (BEVFormer/BLIP-2 style) |
| **LLM** | Qwen2.5-0.5B-Instruct, frozen backbone + LoRA adapter (r=16, α=32) |
| **Visual projection** | Linear layer, `hidden_size → d_llm` |
| **Training** | Single-stage (vision encoder + LoRA trained jointly) — see [Roadmap](#roadmap) for two-stage pretraining |

### Repository structure

BEV-VLM/
├── main.py # Hydra entry point (training)
├── test_inference.py # final inference/evaluation on a full split
├── analyze_results.py # per-question-category metrics aggregation
├── visualize_samples.py # exports sample images (BEV + Q/A) per category
├── make_notebook.py # generates a Jupyter notebook for interactive inspection
├── configs/
│ ├── config.yaml # main Hydra config
│ └── paths/
│ ├── mini_veh.yaml # small subset, for debugging
│ └── veh.yaml # full dataset
└── src/bev/
├── data/
│ └── dataset.py # BEVQADataset, tokenizer, collate functions
├── models/
│ ├── vision.py # Q-Former (BEVQFormerModel) — vision encoder
│ └── model.py # BEVVLM — assembles vision encoder + LLM + LoRA
└── training/
└── train.py # train_epoch, val_loss, evaluate

## Dataset

Based on **NuScenes-QA**, with answers rewritten as natural sentences
(e.g. `"yes"` → `"Yes, there is."`) instead of the original single-word templates.
Questions fall into 5 categories: `exist`, `count`, `object`, `status`, `comparison`.

BEV features (`[128, 200, 200]`) are precomputed and stored as `.pt` tensors.

## Setup

```bash
git clone git@github.com:NazarioPizzicoli/BEV.git
cd BEV
uv sync

# Set your dataset paths (BEV features + question dictionaries) in:
#   configs/paths/veh.yaml
#   configs/paths/mini_veh.yaml
```

Requires: PyTorch, Transformers, PEFT, Hydra, Weights & Biases (run `wandb login` once).

## Usage

### Training

```bash
# Quick debug run on a small subset
uv run main.py paths=mini_veh training.num_epochs=2

# Full training run
uv run main.py paths=veh training.batch_size=2 training.grad_accum=8 training.num_workers=6
```

Key configurable parameters (Hydra CLI overrides): `model.num_queries` (number of visual
tokens), `training.batch_size`, `training.lr`, `data.train_fraction` / `data.val_fraction`
(subsampling for faster iteration), `training.early_stopping_patience`.

### Inference & evaluation

```bash
uv run test_inference.py --ckpt checkpoints/best_vlm.pth --split val --out results.json
uv run analyze_results.py --results results.json
uv run visualize_samples.py --results results.json --bev_dir <path/bev_features_veh/val>
```

## Results

Model: Qwen2.5-0.5B-Instruct + LoRA (r=16), Q-Former with 100 queries, trained on 15% of
the train split / 10% of val (233k → 35k examples), 14 epochs (early stopping), single
RTX 2080 Ti.

| Category | Accuracy | N samples |
|---|---|---|
| Exist | 0.8198 | 16,051 |
| Count | 0.1858 | 11,447 |
| Object | 0.4586 | 10,322 |
| Status | 0.4996 | 9,214 |
| Comparison | 0.6821 | 7,605 |
| **Overall** | **0.5456** | **54,639** |

Observations:
- **Exist** (direct yes/no) and **comparison** questions perform best.
- **Count** is the clear weak point — a known failure mode for VLMs in general, worsened
  here by limited training and a vision encoder trained from scratch.

> Methodological note: these results are not yet directly comparable to BeLLA (reference
> paper), which uses 3–7B LLMs on 4×H100 GPUs with the full dataset. Our setup uses a
> 0.5B LLM on a single consumer GPU with reduced training due to time/hardware
> constraints. See [Roadmap](#roadmap).

## Roadmap

- [ ] **Full training** (100% of the dataset, more epochs) — direct comparison against
      the current reduced baseline
- [ ] **No-vision baseline** (LLM+LoRA only, text-only) — to quantify the actual
      contribution of BEV features
- [ ] **Stage 1 pretraining (BEV-text alignment)**, inspired by BeLLA: preliminary
      vision encoder ↔ LLM alignment on scene descriptions generated from NuScenes +
      CAN bus annotations, prior to VQA finetuning (Stage 2)
- [ ] Extension to **DriveLM** (evaluation via BLEU-4/METEOR/ROUGE-L/CIDEr)

## References

- Liu et al., *Visual Instruction Tuning* (LLaVA), NeurIPS 2023
- Mohan et al., *BeLLA: End-to-End Birds Eye View Large Language Assistant for
  Autonomous Driving*, arXiv:2512.06096, 2025
- Qian et al., *NuScenes-QA: A Multi-modal Visual Question Answering Benchmark for
  Autonomous Driving Scenario*, AAAI 2024

##

1. Prima di iniziare: assicurati di partire dall'ultima versione
cd ~/BEV
git pull origin main

2. Lavora, poi controlla cosa è cambiato
git status
git diff              # per rivedere le modifiche riga per riga, opzionale

3. Aggiungi e committa
git add .
git status             # ultimo controllo: niente di pesante/indesiderato in staging
git commit -m "Messaggio breve e descrittivo (es. 'Add two-stage training support')"

4. Pusha
git push origin main