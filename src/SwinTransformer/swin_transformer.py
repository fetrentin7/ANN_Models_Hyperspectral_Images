import numpy as np

from ANN_Models_Hyperspectral_Images.src.SwinTransformer.model.input_layer import InputLayer

dummy = np.random.rand(100,100,200)
layer_input = InputLayer()
out, pca = layer_input.pca_apply(dummy, 10)

print(out.shape)

out = layer_input.data_shape(out)
print(out.shape)

