import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, patch_size=11, num_classes=16):

        super().__init__()
        self.conv_blocks = nn.Sequential(
            # in_channels x 11 x 11 → 64 x 5 x 5
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # 64 x 5 x 5 → 128 x 3 x 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 128 x 3 x 3 → 256 x 2 x 2
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.flatten_dim = 256 * 3 * 3  # adjust based on your patch size
        # Head 1 — Real/Fake (Sigmoid)
        self.adv_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 1),
            nn.Sigmoid()
        )

        # Head 2 — Class label (Softmax)
        self.cls_head = nn.Sequential(
            nn.Linear(self.flatten_dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.conv_blocks(x)
        features = features.flatten(1)  # (B, flatten_dim)

        validity = self.adv_head(features)  # (B, 1)
        class_pred = self.cls_head(features)  # (B, num_classes)

        return validity, class_pred

