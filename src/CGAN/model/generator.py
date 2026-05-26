import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=16, patch_size=32, in_channels=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, noise_dim)
        self.patch_proj = nn.Linear(in_channels * patch_size * patch_size, noise_dim)
        input_dim = noise_dim + noise_dim  # z + label_emb
        self.fc = nn.Linear(input_dim, 256 * 4 * 4)
        self.conv_blocks = nn.Sequential(
            # 256 x 4 x 4 → 128 x 8 x 8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 128 x 8 x 8 → 64 x 16 x 16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 64 x 16 x 16 → 32 x 32 x 32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 32 x 32 x 32 → in_channels x 32 x 32
            nn.ConvTranspose2d(32, in_channels, kernel_size=1),
            nn.Identity()
        )

    def forward(self, z, labels, x_patch):
        label_emb = self.label_emb(labels)
        combined  = torch.cat([z, label_emb], dim=1)
        out = self.fc(combined)
        out = out.view(-1, 256, 4, 4)
        out = self.conv_blocks(out)
        return out  # (B, 10, 32, 32)