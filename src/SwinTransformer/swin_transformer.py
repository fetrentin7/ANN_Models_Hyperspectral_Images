import numpy as np
import torch
from torch.onnx.ops import attention

from ANN_Models_Hyperspectral_Images.src.SwinTransformer.model.input_layer import InputLayer
from ANN_Models_Hyperspectral_Images.src.SwinTransformer.model.patch_layer import PatchLayer
from ANN_Models_Hyperspectral_Images.src.SwinTransformer.model.patch_merging import PatchMerging

from ANN_Models_Hyperspectral_Images.src.SwinTransformer.model.encoding_block import SwinTransformerBlock


from ANN_Models_Hyperspectral_Images.src.SwinTransformer.attention.window_attention import WindowAttention
from ANN_Models_Hyperspectral_Images.src.SwinTransformer.attention.shifted_window import ShiftedWindowAttention

dummy = np.random.rand(10,10,100)
layer_input = InputLayer()
out, pca = layer_input.pca_apply(dummy, 10)

print(out.shape)

out = layer_input.data_shape(out)

print(out.shape)

patch_layer = PatchLayer(patch_size=(5,5), in_channels=10, embed_dim=96)
out = patch_layer.extract_patch(out)       # [1, 96, 10, 10]
out = patch_layer.embedding_patch(out)

attn = WindowAttention(
    dim=96,
    heads=8,
    head_dim=96 // 8,
    shifted=False,
    window_size=3,
    pos_embedding=False
)

out = attn(out)
print(out.shape)  # should be torch.Size([1, 10

x = torch.randn(1, 1024, 96)

attn = ShiftedWindowAttention(
    dim=96,
    heads=8,
    head_dim=96 // 8,
    window_size=4,
    pos_embedding=False
)

out = attn(x)
print(out.shape)



block = SwinTransformerBlock(dim=96, res=(32,32), win=4, shift=2)
out = block(x)
print(out.shape)

merge = PatchMerging(dim=96)
x = torch.randn(1, 1024, 96)
out = merge(x, 32, 32)
print(out.shape)