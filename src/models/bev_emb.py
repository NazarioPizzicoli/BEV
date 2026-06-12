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
