
import torch
import torch.nn as nn
from model.patch_layer import PatchLayer
from model.encoding_block import SwinTransformerBlock
from model.patch_merging import PatchMerging
from model.output_layer import OutputLayer


class SwinTransformer(nn.Module):
    def __init__(self, in_channels, num_classes, img_size=32, patch_size=4, embed_dim=96, window_size=4):
        super().__init__()

        self.patch_layer = PatchLayer(patch_size=(patch_size, patch_size),
                                      in_channels=in_channels, embed_dim=embed_dim)

        self.res = (img_size // patch_size, img_size // patch_size)

        self.stage1 = nn.Sequential(
            SwinTransformerBlock(dim=embed_dim, res=self.res, win=window_size, shift=0),
            SwinTransformerBlock(dim=embed_dim, res=self.res, win=window_size, shift=window_size // 2)
        )
        self.merge1 = PatchMerging(dim=embed_dim)

        self.stage2 = nn.Sequential(
            SwinTransformerBlock(dim=embed_dim * 2, res=(self.res[0] // 2, self.res[1] // 2), win=window_size, shift=0),
            SwinTransformerBlock(dim=embed_dim * 2, res=(self.res[0] // 2, self.res[1] // 2), win=window_size,
                                 shift=window_size // 2)
        )
        self.merge2 = PatchMerging(dim=embed_dim * 2)

        self.output = OutputLayer(dim=embed_dim * 4, num_classes=num_classes)
    def forward(self, x):
        x = self.patch_layer.extract_patch(x)
        x = self.patch_layer.embedding_patch(x)
        x = self.stage1(x)
        x = self.merge1(x, self.res[0], self.res[1])  # need H, W here
        x = self.stage2(x)
        x = self.merge2(x, self.res[0] // 2, self.res[1] // 2)
        return self.output(x)


#dummy = np.random.rand(10,10,100)
#layer_input = InputLayer()
#out, pca = layer_input.pca_apply(dummy, 10)
#
#print(out.shape)
#
#out = layer_input.data_shape(out)
#
#print(out.shape)
#
#patch_layer = PatchLayer(patch_size=(5,5), in_channels=10, embed_dim=96)
#out = patch_layer.extract_patch(out)       # [1, 96, 10, 10]
#out = patch_layer.embedding_patch(out)
#
#attn = WindowAttention(
#    dim=96,
#    heads=8,
#    head_dim=96 // 8,
#    shifted=False,
#    window_size=3,
#    pos_embedding=False
#)
#
#out = attn(out)
#print(out.shape)  # should be torch.Size([1, 10
#
#x = torch.randn(1, 1024, 96)
#
#attn = ShiftedWindowAttention(
#    dim=96,
#    heads=8,
#    head_dim=96 // 8,
#    window_size=4,
#    pos_embedding=False
#)
#
#out = attn(x)
#print(out.shape)
#
#
#
#block = SwinTransformerBlock(dim=96, res=(32,32), win=4, shift=2)
#out = block(x)
#print(out.shape)
#
#merge = PatchMerging(dim=96)
#x = torch.randn(1, 1024, 96)
#out = merge(x, 32, 32)
#print(out.shape)


