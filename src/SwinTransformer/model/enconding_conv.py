import torch
import torch.nn as nn

from ANN_Models_Hyperspectral_Images.src.SwinTransformer.attention.window_attention import WindowAttention
from ANN_Models_Hyperspectral_Images.src.SwinTransformer.attention.shifted_window import ShiftedWindowAttention
from ANN_Models_Hyperspectral_Images.src.SwinTransformer.model.swin_conv_layer import LPU, IRFFN


class SwinTransformerBlockConv(nn.Module):

    def __init__(self, dim, res, win, shift, lpu_mode=None, lpu_kernel=3, use_irffn=False):
        super().__init__()
        self.dim = dim
        self.res = res
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.lpu = LPU(dim, mode=lpu_mode, kernel=lpu_kernel) if lpu_mode else None

        self.use_irffn = use_irffn
        if use_irffn:
            self.mlp = IRFFN(dim, ratio=4.0)
        else:
            self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim),nn.GELU(),nn.Linear(4 * dim, dim))

        self.shift = shift
        H, W = res
        if shift > 0:
            self.mask = self.create_mask(H, W, win, shift)
            self.attn = ShiftedWindowAttention(dim=dim, heads=8, head_dim=dim // 8,
                                               window_size=win, pos_embedding=False)
        else:
            self.mask = None
            self.attn = WindowAttention(dim=dim, heads=8, head_dim=dim // 8,
                                        shifted=False, window_size=win, pos_embedding=False)

    def create_mask(self, H, W, win, shift):

        img_mask = torch.zeros((1, H, W, 1))
        count = 0

        for h in (slice(0, -win), slice(-win, -shift), slice(-shift, None)):
            for w in (slice(0, -win), slice(-win, -shift), slice(-shift, None)):
                img_mask[:, h, w, :] = count
                count += 1

        mask_windows = img_mask.reshape(1, H // win, win, W // win, win, 1)
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5)
        mask_windows = mask_windows.reshape(-1, win * win)

        # compare labels
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

        return attn_mask

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.res

        if self.lpu is not None:
            x = self.lpu(x, H, W)  # CMT Eq. 7

        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + residual  # CMT Eq. 8

        y = self.norm2(x)
        x = x + (self.mlp(y, H, W) if self.use_irffn else self.mlp(y))  # CMT Eq. 9
        return x

B, H, W, C = 2, 8, 8, 96
x = torch.randn(B, H * W, C)

lpu = LPU(C)
out = lpu(x, H, W)
print(out.shape, torch.allclose(out, x))   # esperado: (2, 64, 96)  False

irffn = IRFFN(C)
print(irffn(x, H, W).shape)
