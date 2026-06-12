from torch import nn
from src.utils.bev_emb import BEVAdapter
from src.utils.text_emb import TextAdapter
from src.utils.mca import MCALayer
from src.utils.head import OutputHead


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
