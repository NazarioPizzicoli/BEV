# BEV-VQA: Multimodal SOTA Framework for Autonomous Driving

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![WandB](https://img.shields.io/badge/Weights%20%26%20Biases-enabled-orange.svg)](https://wandb.ai/)

Questo progetto implementa un framework avanzato di **Visual Question Answering (VQA)** per la guida autonoma. Integra feature spaziali **Bird's Eye View (BEV)** pre-fused con elaborazione del linguaggio naturale (GloVe) per fornire risposte semantiche sulle scene stradali NuScenes.

## 📂 Struttura della Repository

```text
BEV/
├── bev/                    # 🐍 Ambiente virtuale Python (venv) attivo
├── configs/                # ⚙️ Gestione configurazioni (Hydra)
│   ├── paths/              # Mapping dei path locali (mini vs default)
│   └── config.yaml         # Iperparametri e scelta del modello
├── src/bev/                # 🧠 Core Logic
│   ├── data/               # Dataset NuScenes BEV-VQA
│   ├── models/             # Architetture SOTA (ResNet, ViT, BEVFormer)
│   ├── training/           # Engine di training (AMP, Early Stopping)
│   └── utils/              # Utility (GloVe, Text decoding)
├── scripts/                # 📊 Analisi e metriche post-training
│   ├── evaluate.py         # Genera report e visualizzazioni BEV
│   └── metrics.py          # Confusion Matrix e Precision/Recall/F1
├── checkpoints/            # 💾 Salvataggio pesi (.pth)
├── notebooks/              # 📓 Pipeline interattive e visualizzazione
└── main.py                 # 🚀 Entry point universale
```

## 🧠 Modelli e Architetture BEV Supportate

È possibile scegliere tra 4 diverse filosofie di encoding spaziale tramite Hydra:

1.  **`linear` (Baseline)**: Proiezione semplice con pooling adattivo.
2.  **`cnn` (ResNet50 + FPN)**: Architettura ispirata a **BEVDet/BEVFusion** con backbone profondo e feature multi-scala.
3.  **`vit` (BEV-MAE style)**: Vision Transformer a 6 layer con **Random Masking** (50%) per una robustezza estrema alle occlusioni.
4.  **`former` (BEVFormer style)**: Basato su **Learnable Queries** e Cross-Attention spaziale per interrogare dinamicamente la mappa BEV.

## 🛠️ Setup e Installazione

### Ambiente Python
L'ambiente consigliato è situato nella cartella `bev/`. Per attivarlo:
```bash
source bev/bin/activate
```

### Dipendenze Critiche
Per utilizzare le architetture SOTA e gli strumenti di analisi, assicurati di avere installato:
```bash
pip install torchvision wandb scikit-learn seaborn
```

## 🚀 Training e Configurazione

Utilizziamo **Hydra** per gestire gli esperimenti senza modificare il codice.

```bash
# Esempio: Training con BEVFormer sul dataset mini
python main.py model.type=former paths=mini training.batch_size=16

# Esempio: Training con CNN (ResNet50) e pesi ImageNet
python main.py model.type=cnn training.lr=1e-4
```

## 📊 Analisi e Valutazione

Dopo il training, puoi generare report completi e visualizzare cosa "vede" il modello:

### Valutazione e Visualizzazione BEV
```bash
python scripts/evaluate.py \
    --run_dir outputs/mini.yaml/DATA/ORA \
    --checkpoint checkpoints/best_model_cnn.pth
```
Genera: `metrics_plot.png`, `eval_results.json` e **`sample_eval.png`** (Mappa BEV con Q&A).

### Metriche Avanzate (Notebook)
Utilizza `scripts/metrics.py` all'interno del notebook `pipeline_01_visual.ipynb` per generare la **Matrice di Confusione** multiclasse e il report dettagliato (Precision, Recall, F1) per ogni categoria di risposta.

## 📈 Weights & Biases (WandB)
Il framework è integrato con WandB. Durante il training vengono loggati:
- Loss e Accuracy (Train/Val) in tempo reale.
- **Visual Predictions**: Immagini delle mappe BEV con predizioni caricate periodicamente sul dashboard.
- Matrice di confusione interattiva a fine run.



### Componenti:
- uv.lock: Serve a garantire la riproducibilità matematica del tuo ambiente di sviluppo.
- pyproject.toml: è il file di configurazione standard moderno per l'ecosistema Python.
- Makefile: automazione dei comandi Docker, make build per compilare l'immagine Docker, make run per far partire il container, make stop per spegnerlo e make clean per fare tabula rasa.
