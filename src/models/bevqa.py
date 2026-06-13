from torch import nn
from src.models.bev_emb import BEVAdapter, BEVAdapterCNN, BEVAdapterViT
from src.models.text_emb import TextAdapter
from src.models.mca import MCALayer
from src.models.head import OutputHead


class BEVQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.bev_adapter = BEVAdapter()
        self.text_adapter = TextAdapter()
        self.mca = MCALayer()
        self.head = OutputHead()

    def forward(self, bev, text):
        bev = self.bev_adapter(bev)  # [B, 400, 512]
        text = self.text_adapter(text)  # [B, 35, 512]
        bev, text = self.mca(bev, text)
        return self.head(text)  # [B,30]

class BEVQA_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bev_adapter = BEVAdapterCNN()
        self.text_adapter = TextAdapter()
        self.mca = MCALayer()
        self.head = OutputHead()

    def forward(self, bev, text):
        bev = self.bev_adapter(bev)  # [B, 100, 512]
        text = self.text_adapter(text)  # [B, 35, 512]
        bev, text = self.mca(bev, text)
        return self.head(text)  # [B,30]

class BEVQA_ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bev_adapter = BEVAdapterViT()
        self.text_proj   = TextAdapter()
        self.mca         = MCALayer()
        self.head        = OutputHead()
    
    def forward(self, bev, text):
        bev  = self.bev_adapter(bev)    # [B,100,512]
        text = self.text_proj(text)     # [B,35,512]
        bev, text = self.mca(bev, text)
        return self.head(text)          # [B,30]