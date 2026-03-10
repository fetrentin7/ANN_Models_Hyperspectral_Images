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

from CNN import CNN2D

# Load the image
DATA_PATH = np.load(
    r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/indianpinearray.npy"
)

using_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', using_gpu)


labels = np.load(
    r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/ipgt.npy"
)
#check label
print(DATA_PATH.shape)
print(labels.shape)

def pca_apply(X,  n_components):
    # reshapes the numpy array into a new one and the tuple is defining the array dimension (2)
    # -1 a placeholder to automatically calculate the required size for this dimension based on the total number of elements in the array and the size of the other dimension.
    new_x = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components, whiten=True)
    x_value = pca.fit_transform(new_x)
    new_X = np.reshape(x_value, (X.shape[0], X.shape[1], n_components))
    return new_X, pca

#apply pca

pca_data, pca_model = pca_apply(DATA_PATH, 30)
print(type(pca_data))
print(len(pca_data) if isinstance(pca_data, tuple) else "not tuple")
print(pca_data.shape if hasattr(pca_data, "shape") else "no shape")


#creating the training patches
def create_patches(x, y, size):
    margin = size//2
    padded_x = np.pad(x, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
    labels = []
    list = []
    for i in range(margin, padded_x.shape[0] - margin):
        for j in range(margin, padded_x.shape[1] - margin):
            patch = np.zeros((size, size, padded_x.shape[2]))
            patch = padded_x[i-margin:i+margin+1, j-margin:j+margin+1, :]

            list.append(patch)
            labels.append(y[i - margin, j - margin])

    return np.array(list), np.array(labels)

PATCH_SIZE = 15
patches, patch_label = create_patches(pca_data, labels, PATCH_SIZE)
#filtering non-zero labels, and sifht to c-1
non_zero = patch_label > 0
patches = patches[non_zero]
patch_label = patch_label[non_zero] - 1

x_training, x_test, y_train, y_test = train_test_split(patches, patch_label, test_size=0.2, random_state=42)

#converting to pytorch tensors --> switch to channelss first (N,C,H,W)

x_training = np.transpose(x_training, (0,3,1,2)).astype(np.float32)
x_test = np.transpose(x_test, (0,3,1,2)).astype(np.float32)
y_training = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

train_ds = TensorDataset(torch.from_numpy(x_training), torch.from_numpy(y_training))
test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

num_classes = int(np.unique(patch_label).shape[0])

in_channel = pca_data.shape[-1]
model = CNN2D(channels=in_channel, classes=num_classes).to(using_gpu)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 50

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(using_gpu)
            yb = yb.to(using_gpu)

            logits = model(xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(dim = 1)
            correct+= (preds == yb).sum().item()
            total += yb.size(0)

    return loss_sum/total, correct/total

train_history = {'loss': [], 'val_loss':[], 'val_acc':[]}


for epoch in range(1, EPOCHS + 1):

    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(using_gpu)
        yb = yb.to(using_gpu)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss/ len(train_loader.dataset)
    val_loss, val_acc = evaluate(test_loader)

    train_history['loss'].append(train_loss)
    train_history['val_loss'].append(val_loss)
    train_history['val_acc'].append(val_acc)

    print(f'{epoch:02d}/{EPOCHS} |train_loss= {train_loss: .4f} | val_loss={val_loss: .4f} | val_acc={val_acc: .4f} ')

def results(data_pca, labels, model, patch_size):
    model.eval()
    margin = patch_size//2
    padded = np.pad(data_pca, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
    H,W,C  = data_pca.shape

    patches = []

    for i in range(margin, padded.shape[0] - margin):
        for j in range(margin, padded.shape[1] - margin):
            px = padded[i - margin: i+margin+1, j - margin: j+margin+1,:]
            patches.append(px)

    patches = np.array(patches, dtype=np.float32)

    #converting to pytorch
    patches_t = torch.from_numpy(np.transpose(patches, (0,3,1,2)))
    preds_all = []

    with torch.no_grad():
        for start in range(0, patches_t.shape[0], 64):
            batch = patches_t[start:start+64].to(using_gpu)
            logits = model(batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            preds_all.append(preds)

    predicted_labels = np.concatenate(preds_all, axis=0).reshape(labels.shape)
    predicted_labels_masked = predicted_labels * (labels != 0)

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.title('Ground Truth')
    plt.imshow(labels, cmap='jet')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("predicted (Full)")
    plt.imshow(predicted_labels, cmap='jet')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title('Predicted [masked]')
    plt.imshow(predicted_labels_masked, cmap='jet')
    plt.axis('off')

    plt.show()

results(pca_data, labels, model, PATCH_SIZE)
