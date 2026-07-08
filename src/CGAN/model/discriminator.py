import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, in_channels=50, patch_size=32, num_classes=16):
        super().__init__()

        # Doubled feature maps: 128 -> 256 -> 512
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),  # Added Batch Norm here to stabilize the wider network
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d((8, 8))  # Forces output to 8x8
        )

        # 512 channels * 8 height * 8 width = 32768
        self.flatten_dim = 512 * 8 * 8

        # Head 1 — Real/Fake (Sigmoid)
        self.adv_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 1),
            nn.Sigmoid()
        )

        # Head 2 — Class label (Softmax
        self.cls_head = nn.Sequential(
            nn.Linear(self.flatten_dim, num_classes),
        )

    def forward(self, x):
        features = self.conv_blocks(x)
        features = features.flatten(1)  # (B, 32768)

        validity = self.adv_head(features)  # (B, 1)
        class_pred = self.cls_head(features)  # (B, num_classes)

        return validity, class_pred