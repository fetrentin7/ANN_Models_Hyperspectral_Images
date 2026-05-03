import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=16, patch_size=11, in_channels=3):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, noise_dim)
        # Project X (flattened patch) to same space
        self.patch_proj = nn.Linear(in_channels * patch_size * patch_size, noise_dim)
        # Combined dim = noise_dim * 3 (Z + C_emb + X_proj)
        self.fc = nn.Linear(noise_dim * 3, 256 * 4 * 4)
        self.conv_blocks = nn.Sequential(
            # 256 x 4 x 4 → 128 x 8 x 8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 128 x 8 x 8 →  64 x 11 x 11 (match patch size)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 64 x 11 x 11 → in_channels x 11 x 11
            nn.ConvTranspose2d(64, in_channels, kernel_size=1),
            nn.Tanh()  # output normalized like real data
        )

    def forward(self, z, labels, x_patch):
        # Embed and project each input
        label_emb = self.label_emb(labels)  # (B, noise_dim)
        patch_emb = self.patch_proj(x_patch.flatten(1))  # (B, noise_dim)
        # Concatenate all conditioning info
        combined = torch.cat([z, label_emb, patch_emb], dim=1)  # (B, noise_dim*3)
        out = self.fc(combined)
        out = out.view(-1, 256, 4, 4)  # reshape to spatial
        out = self.conv_blocks(out)  # (B, in_channels, patch_H, patch_W)
        return out