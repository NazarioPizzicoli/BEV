from torch import nn


class OutputHead(nn.Module):
    def __init__(self, d_model=512, num_classes=30):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, text):  # [B, 35, 512]
        x = text.mean(dim=1)  # [B, 512]
        return self.fc(x)  # [B, 30] 30 come il numero di risposte possibili
