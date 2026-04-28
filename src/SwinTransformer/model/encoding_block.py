import torch
import torch.nn as nn

from ANN_Models_Hyperspectral_Images.src.SwinTransformer.attention.window_attention import WindowAttention
from ANN_Models_Hyperspectral_Images.src.SwinTransformer.attention.shifted_window import ShiftedWindowAttention

class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, res, win, shift):
        super().__init__()
        self.dim = dim
        self.res = res
        self.norm1 = nn.LayerNorm(dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )

        self.shift = shift
        H,W = res
        if shift > 0:
            self.mask = self.create_mask(H, W, win, shift)  # for blocking wrapped patches
            self.attn = ShiftedWindowAttention(dim=dim, heads=8, head_dim=dim // 8,
                                               window_size=win, pos_embedding=False)
        else:
            self.mask = None
            self.attn = WindowAttention(dim=dim, heads=8, head_dim=dim // 8,
                                        shifted=False, window_size=win, pos_embedding=False)


    def create_mask(self, H, W,win, shift):

        img_mask = torch.zeros((1,H,W,1))
        count = 0

        for h in (slice(0,-win), slice(-win, -shift), slice(-shift, None)):
            for w in (slice(0, -win), slice(-win, -shift), slice(-shift, None)):
                img_mask[:,h,w,:] = count
                count+=1

        mask_windows = img_mask.reshape(1, H // win, win, W // win, win, 1)
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5)
        mask_windows = mask_windows.reshape(-1, win * win)

        #compare labels
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

        return attn_mask

    def forward(self, x):
        B, L, C = x.shape

        H,W = self.res

        residual = x
        x = self.norm1(x)

        x = self.attn(x)
        x = x + residual  # first residual add

        x = x + self.mlp(self.norm2(x))  # second residual add
        return x