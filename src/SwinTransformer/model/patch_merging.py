import torch.nn as nn
import torch

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)


    def forward(self, x, H, W):
        B,L,C = x.shape
        x = x.view(B,H,W,C)
        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-righ
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4*dim]
        x = x.reshape(B, -1, 4 * self.dim)  # flatten spatial
        x = self.norm(x)
        x = self.reduction(x)
        return x