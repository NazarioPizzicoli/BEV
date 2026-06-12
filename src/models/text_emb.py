from torch import nn


class TextAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(300, 512)

    def forward(self, x):
        return self.proj(x)  # [B,35,300] to match BEV emb [B,35,512]
