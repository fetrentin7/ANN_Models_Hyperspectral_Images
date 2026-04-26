import torch.nn as nn
class OutputLayer(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dense = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.dense(x)
        return x