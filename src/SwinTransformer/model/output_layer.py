import torch.nn as nn
class OutputLayer:
    def __init__(self, dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dense = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling
        x = self.dense(x)  # dense layer
        return x  # softmax applied in loss function