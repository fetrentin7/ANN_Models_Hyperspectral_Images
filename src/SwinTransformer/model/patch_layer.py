import torch.nn as nn
import torch
class PatchLayer(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_sizeX = patch_size[0]
        self.patch_sizeY = patch_size[1]
        self.emb = embed_dim
        self.in_channels = in_channels
        self.projection = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.emb,
            kernel_size=(self.patch_sizeX, self.patch_sizeY),
            stride=(self.patch_sizeX, self.patch_sizeY)
        )


    def extract_patch(self, x):
        return self.projection(x)


    def embedding_patch(self, patches):
        flat_dimension = patches.flatten(2)
        tokens = flat_dimension.permute(0,2,1)
        return tokens