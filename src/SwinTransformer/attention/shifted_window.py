import torch
import torch.nn as nn
from .window_attention import WindowAttention

class ShiftedWindowAttention(WindowAttention):
    def __init__(self, dim, heads, head_dim, window_size, pos_embedding):
        super().__init__(dim, heads, head_dim, shifted=True, window_size=window_size, pos_embedding=pos_embedding)

    def forward(self, x):
        B, seq, dim = x.shape #batch size
        H = W = int(seq**0.5)

        x = x.reshape(B, H, W, dim)
        x = torch.roll(x, shifts=(-self.window_size // 2, -self.window_size // 2), dims=(1, 2))


        x = x.reshape(
            B,
            H // self.window_size,
            self.window_size,
            W // self.window_size,
            self.window_size,
            dim
        )

        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, self.window_size * self.window_size, dim)

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(-1,
                      self.window_size * self.window_size,
                      self.heads, dim // self.heads).permute(0, 2, 1, 3)
        k = k.reshape(-1,
                      self.window_size * self.window_size,
                      self.heads, dim // self.heads).permute(0, 2, 1, 3)

        v = v.reshape(-1, self.window_size * self.window_size,
                      self.heads, dim // self.heads).permute(0, 2, 1, 3)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)

        out = attn @ v

        # merge heads
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(-1, self.window_size * self.window_size, dim)
        # proj
        out = self.proj(out)

        # merge windows
        num_windows_h = H // self.window_size
        num_windows_w = W // self.window_size

        out = out.reshape(
            B,
            num_windows_h,
            num_windows_w,
            self.window_size,
            self.window_size,
            dim
        )

        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, dim)
        out = torch.roll(out, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))  # add this

        out = out.reshape(B, H * W, dim)
        return out