import spectral
import os
import numpy as np
import matplotlib.pyplot as plt
from spectral import open_image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import scipy.io as sci
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Load the image
DATA_PATH = np.load(
    r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/indianpinearray.npy"
)

using_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', using_gpu)

#check label

def pca_apply(X,  n_components):
    # reshapes the numpy array into a new one and the tuple is defining the array dimension (2)
    # -1 a placeholder to automatically calculate the required size for this dimension based on the total number of elements in the array and the size of the other dimension.
    new_x = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components, whiten=True)
    x_value = pca.fit_transform(new_x)
    new_X = np.reshape(x_value, (X.shape[0], X.shape[1], n_components))
    return new_X, pca


#apply pca
pca_data = pca_apply(DATA_PATH, 30)

#creating the training patches
def create_patches(x, y, size):
    margin = size//2
    padded_x = np.pad((margin, margin), (margin, margin), (0,0), mode='constant')
    list = []

    for i in range(margin, padded_x.shape[0] - margin):
        for j in range(margin, padded_x[1] - margin):
            patch = np.zeros((size, size, padded_x.shape[2]))
            for di in range(size):
                for dj in range(size):
                    for dc in range(padded_x.shape[2]):
                        patch[di, dj, dc] = padded_x[i - margin + di, j - margin + dj, dc]