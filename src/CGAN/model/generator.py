import torch.nn as nn
import torch


class Generator(nn.Module):
    # Increased noise_dim to 512 to give the network more complex random variables
    def __init__(self, noise_dim=512, num_classes=16, patch_size=32, in_channels=50):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, noise_dim)

        input_dim = noise_dim * 2  # z + label_emb

        # Massively increased base feature maps: 1024 (was 256)
        self.fc = nn.Linear(input_dim, 1024 * 4 * 4)

        self.conv_blocks = nn.Sequential(
            # 1024 x 4 x 4 → 512 x 8 x 8
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 512 x 8 x 8 → 256 x 16 x 16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 256 x 16 x 16 → 128 x 32 x 32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 128 x 32 x 32 → in_channels x 32 x 32
            nn.ConvTranspose2d(128, in_channels, kernel_size=1),

            # Bound the output! Use Tanh if data is [-1, 1], Sigmoid if [0, 1]
            nn.Tanh()
        )

    # Kept x_patch in the signature so your training script G(z, yb, xb) doesn't crash
    def forward(self, z, labels, x_patch=None):
        label_emb = self.label_emb(labels)
        combined = torch.cat([z, label_emb], dim=1)
        out = self.fc(combined)
        out = out.view(-1, 1024, 4, 4)
        out = self.conv_blocks(out)
        return out  # (B, 50, 32, 32)