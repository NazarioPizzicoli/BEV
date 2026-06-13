from pathlib import Path

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import rootutils

rootutils.setup_root(Path(".").resolve(), indicator=".project-root", pythonpath=True)

from src.models.bev_emb import BEVAdapterViT
from src.models.bevqa import BEVQA_ViT
from src.data.bevqa_dataset import BEVQADataset
from src.models.head import OutputHead
from src.models.mca import MCALayer
from src.models.text_emb import TextAdapter
from src.training.train import train_epoch, val_epoch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device:{device}")

ROOT = Path("/home/robesafe-sandra/BEV")
DISK = Path("/media/robesafe-sandra/Datos")
DIR = DISK / "dataset"
DICT_DIR = DIR / "dict"
FEAT_DIR = DIR / "bev_features"

GLOVE = ROOT/ "glove.6B/glove.6B.300d.txt"

train_dataset = BEVQADataset(
    bev_dir=FEAT_DIR / "train",
    json_path=DICT_DIR / "NuScenes_train_questions.json",
    glove=GLOVE
)
val_dataset = BEVQADataset(
    bev_dir=FEAT_DIR / "val",
    json_path=DICT_DIR / "NuScenes_val_questions.json",
    glove=GLOVE
)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

model = BEVQA_ViT().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 20

for epoch in range(num_epochs):
    print("Training:...")
    tr_loss, tr_acc = train_epoch(model, train_dataloader, optimizer, criterion, device)
    val_loss, val_acc = val_epoch(model, val_dataloader, criterion, device)
    print(f"Epoch {epoch+1:02d}/20 | Train Loss: {tr_loss:.4f} - Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), ROOT / "best_model_vit.pth")
        print(f"  → Saved best model (val_acc={val_acc:.4f})")