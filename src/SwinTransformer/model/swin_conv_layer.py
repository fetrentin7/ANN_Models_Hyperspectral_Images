import torch.nn as nn

import torch.nn as nn

class LPU(nn.Module):
    def __init__(self, dim, mode="2d", kernel=3):
        super().__init__()
        self.mode = mode
        if mode == "2d":
            self.dw = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        elif mode == "1d":
            self.dw = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        else:
            raise ValueError(f"invalid mode: {mode}")

    def forward(self, x, h, w):
        b, l, c = x.shape
        if self.mode == "2d":
            y = x.transpose(1, 2).reshape(b, c, h, w)
            y = self.dw(y).flatten(2).transpose(1, 2)
        else:
            y = x.transpose(1, 2)
            y = self.dw(y).transpose(1, 2)
        return x + y

class IRFFN(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.bn = nn.BatchNorm2d(hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x, h, w):
        x = self.act(self.fc1(x))
        b, l, c = x.shape
        y = x.transpose(1, 2).reshape(b, c, h, w)
        y = self.act(self.bn(self.dw(y))).flatten(2).transpose(1, 2)
        x = x + y
        return self.fc2(x)