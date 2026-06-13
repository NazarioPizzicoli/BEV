import torch
from torch import nn
# [B, 128, 200, 200] -> AvgPool(20,20) -> [B,128,20,20] -> reshape -> [B, 400, 128] -> Linear(128->512) -> [4,400,512]


class BEVAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((20, 20))
        self.proj = nn.Linear(128, 512)

    def forward(self, x):
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class BEVAdapterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # [B,256,100,100]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # [B,512,50,50]
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=5, padding=1), # [B,512,10,10]
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class BEVAdapterViT(nn.Module):
    def __init__(self, patch_size=20, d_model=512):
        super().__init__()
        patch_dim = 128 * patch_size * patch_size  # 128*20*20 = 51200
        self.patch_size = patch_size
        
        # Linear projection patch → d_model
        self.proj = nn.Linear(patch_dim, d_model)
        
        # Positional encoding learnable
        self.pos_enc = nn.Parameter(torch.randn(1, 100, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
    
    def forward(self, x):  # [B,128,200,200]
        B, C, H, W = x.shape
        p = self.patch_size
        
        # Divide in patch
        x = x.reshape(B, C, H//p, p, W//p, p)   # [B,128,10,20,10,20]
        x = x.permute(0, 2, 4, 1, 3, 5)          # [B,10,10,128,20,20]
        x = x.reshape(B, 100, -1)                 # [B,100,51200]
        
        x = self.proj(x)                          # [B,100,512]
        x = x + self.pos_enc                      # [B,100,512]
        x = self.transformer(x)                   # [B,100,512]
        return x