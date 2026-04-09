import torch.nn as nn
import torch
class PatchLayer:
    def __init__(self, patch_size, in_channels, embed_dim):

        self.patch_sizeX = patch_size[0]
        self.patch_sizeY = patch_size[1]
        self.emb = embed_dim
        self.in_channels = in_channels
        nn.Conv2d(
            in_channels = self.in_channels,  #spectral bands
            out_channels = embed_dim,
            kernel_size = self.patch_sizeX,  # patch size from article
            stride = 2 # moves 2 pixels = no overlap, halves dimensions
        )

        self.projection = nn.Conv2d()
    def extract_patch(self, x):
        return self.projection(x)


    def embedding_patch(self, patches):
        flat_dimension = patches.flatten(2)
        tokens = flat_dimension.permute(0,2,1)
        return tokens