import numpy as np
from sklearn.decomposition import PCA
import torch

class InputLayer:

    def pca_apply(self, X, n_components):
        new_x = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components, whiten=True)
        x_value = pca.fit_transform(new_x)
        new_X = np.reshape(x_value, (X.shape[0], X.shape[1], n_components))
        return new_X, pca

    def data_shape(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        X = X.permute(2,0,1) # [H, W, Spectral] - > [Spectral, H, W]
        X = X.unsqueeze(0) # adiciona dim de batch ->[S, H, W] → [1, S, H, W]
        return X

