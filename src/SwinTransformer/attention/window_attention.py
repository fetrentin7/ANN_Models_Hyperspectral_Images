import torch
import torch.nn as nn
class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, pos_embedding):
        super().__init__()

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = pos_embedding
        self.shifted = shifted

        self.to_qkv = nn.Linear(dim, dim*3) #concatenating side by side in one output
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, seq, dim = x.shape #batch size
        H = W = int(seq**0.5)

        x = x.reshape(B, H, W, dim)

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

        #merge heads
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(-1, self.window_size * self.window_size, dim)
        out = self.proj(out)
        #proj
        out = self.proj(out)

        #merge windows
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
        out = out.reshape(B, H * W, dim)
        return out
