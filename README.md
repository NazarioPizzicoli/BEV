# BEV-VLM: Visual Question Answering su Bird's-Eye-View per Guida Autonoma

Modello Vision-Language generativo (image-text-to-text) che risponde a domande in linguaggio
naturale su scene di guida autonoma, a partire da feature Bird's-Eye-View (BEV) invece che
da immagini camera dirette.

## Panoramica

Il progetto segue la ricetta **LLaVA** (visual instruction tuning): una feature map BEV
viene incapsulata in "visual token" tramite un encoder attention-based (Q-Former), proiettata
nello spazio di embedding di un LLM (Qwen2.5), e concatenata al prompt testuale. Il LLM,
con LoRA, genera risposte naturali (non risposte a singola parola/template).

BEV [B,128,200,200] --(Q-Former)--> [B,N,hidden] --(proj. lineare)--> [B,N,d_llm]
--> spliced sui placeholder <|bev|> nel prompt --> Qwen2.5 (frozen + LoRA)
--> risposta generata in linguaggio naturale

Il progetto è ispirato a [BeLLA (Mohan et al., 2025)](https://arxiv.org/abs/2512.06096),
di cui riprendiamo tecniche selezionate (in particolare il pretraining di allineamento
BEV-testo) adattandole alla nostra architettura e al nostro caso d'uso (feature BEV
pre-calcolate, non immagini multi-camera raw).

## Architettura

| Componente | Implementazione |
|---|---|
| **Vision encoder** | Q-Former: query learnable + self/cross-attention verso la feature BEV flattenata (stile BEVFormer/BLIP-2) |
| **LLM** | Qwen2.5-0.5B-Instruct, frozen + adapter LoRA (r=16, α=32) |
| **Proiezione visiva** | Lineare, `hidden_size → d_llm` |
| **Training** | Stadio singolo (LoRA + vision encoder allenati insieme) — vedi [Roadmap](#roadmap) per doppio stadio |

### Struttura del repository
BEV-VLM/
├── main.py # entry point Hydra (training)
├── test_inference.py # inferenza/valutazione finale su uno split completo
├── analyze_results.py # aggregazione metriche per categoria di domanda
├── visualize_samples.py # esporta esempi (BEV + Q/A) come immagini, per categoria
├── make_notebook.py # genera un notebook Jupyter per ispezione interattiva
├── configs/
│ ├── config.yaml # config principale (Hydra)
│ └── paths/
│ ├── mini_veh.yaml # subset piccolo, per debug
│ └── veh.yaml # dataset completo
└── src/bev/
├── data/
│ └── dataset.py # BEVQADataset, tokenizer, collate functions
├── models/
│ ├── vision.py # Q-Former (BEVQFormerModel) — vision encoder
│ └── model.py # BEVVLM — assembla vision encoder + LLM + LoRA
└── training/
└── train.py # train_epoch, val_loss, evaluate
## Dataset

Basato su **NuScenes-QA**, con risposte riformulate in linguaggio naturale
(es. `"yes"` → `"Yes, there is."`) invece dei template a singola parola originali.
Le domande sono categorizzate in 5 tipi: `exist`, `count`, `object`, `status`, `comparison`.

Le feature BEV (`[128, 200, 200]`) sono precalcolate e salvate come tensori `.pt`.

## Setup

```bash
git clone <repo-url>
cd BEV-VLM
uv sync   # o: pip install -r requirements.txt

# Configura i path del dataset (feature BEV + dizionari domande) in:
#   configs/paths/veh.yaml
#   configs/paths/mini_veh.yaml
```

Servono: PyTorch, Transformers, PEFT, Hydra, Weights & Biases (login già effettuato
con `wandb login`).

## Uso

### Training

```bash
# Debug rapido su subset piccolo
uv run main.py paths=mini_veh training.num_epochs=2

# Training completo
uv run main.py paths=veh training.batch_size=2 training.grad_accum=8 training.num_workers=6
```

Parametri principali configurabili via CLI Hydra (override), tra cui:
`model.num_queries` (n. di visual token), `training.batch_size`, `training.lr`,
`data.train_fraction`/`data.val_fraction` (sottocampionamento per training più rapidi),
`training.early_stopping_patience`.

### Inferenza e valutazione

```bash
uv run test_inference.py --ckpt checkpoints/best_vlm.pth --split val --out results.json
uv run analyze_results.py --results results.json
uv run visualize_samples.py --results results.json --bev_dir <path/bev_features_veh/val>
```

## Risultati

Modello: Qwen2.5-0.5B-Instruct + LoRA (r=16), Q-Former con 100 query, training su
15% del train set / 10% del val set (233k → 35k esempi), 14 epoche (early stopping),
RTX 2080 Ti.

| Categoria | Accuracy | N esempi |
|---|---|---|
| Exist | 0.8198 | 16.051 |
| Count | 0.1858 | 11.447 |
| Object | 0.4586 | 10.322 |
| Status | 0.4996 | 9.214 |
| Comparison | 0.6821 | 7.605 |
| **Overall** | **0.5456** | **54.639** |

Osservazioni:
- Le domande di **esistenza** (yes/no dirette) e **comparazione** hanno le performance
  migliori.
- Il **conteggio** è il punto debole più marcato — comportamento noto in letteratura per
  i VLM in generale, aggravato qui da poco training e vision encoder inizializzato da zero.

> Nota metodologica: questi risultati non sono ancora direttamente comparabili con
> BeLLA (paper di riferimento), che usa LLM da 3-7B su 4×H100 con dataset completo;
> il nostro setup usa un LLM 0.5B su singola GPU consumer con training ridotto per
> vincoli di tempo/hardware. Vedi [Roadmap](#roadmap).

## Roadmap

- [ ] **Training completo** (100% dataset, più epoche) — confronto diretto con la
      baseline attuale ridotta
- [ ] **Baseline senza vision encoder** (solo LLM+LoRA, text-only) — per quantificare
      l'apporto reale delle feature BEV
- [ ] **Pretraining Stage 1 (BEV-text alignment)**, ispirato a BeLLA: allineamento
      preliminare vision encoder ↔ LLM su descrizioni di scena generate da annotazioni
      NuScenes + CAN bus, prima del finetuning VQA (Stage 2)
- [ ] Estensione a **DriveLM** (valutazione con BLEU-4/METEOR/ROUGE-L/CIDEr)

## Riferimenti

- Liu et al., *Visual Instruction Tuning* (LLaVA), NeurIPS 2023
- Mohan et al., *BeLLA: End-to-End Birds Eye View Large Language Assistant for
  Autonomous Driving*, arXiv:2512.06096, 2025
- Qian et al., *NuScenes-QA: A Multi-modal Visual Question Answering Benchmark for
  Autonomous Driving Scenario*, AAAI 2024
  