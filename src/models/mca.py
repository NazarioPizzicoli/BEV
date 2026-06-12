from torch import nn


class MCALayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        # Self-Attention
        self.sa_bev = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.sa_text = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Cross-Attention
        self.ca = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Layer Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, bev, text):
        # bev: [B, 400, 512]
        # text: [B, 35, 512]
        # Self-Attention
        bev = self.norm1(bev + self.sa_bev(bev, bev, bev)[0])
        text = self.norm2(text + self.sa_text(text, text, text)[0])

        # Cross-Attention
        text = self.norm3(text + self.ca(text, bev, bev)[0])

        return bev, text
