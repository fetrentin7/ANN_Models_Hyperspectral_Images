import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.ndimage import binary_dilation
from CNN import CNN2D
import torch.nn.functional as F

from testing import checkerboard_split, setup_device, load_data, pca_apply, create_patches, results


using_gpu = setup_device()

DATA_PATH = r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/indianpinearray.npy"
LABEL_PATH = r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/IPgt.npy"

# 2. Pass the strings into your function to load the arrays
data, labels = load_data(DATA_PATH, LABEL_PATH)
print(f"Original Data Shape: {data.shape} | Labels Shape: {labels.shape}")

# Apply PCA
COMPONENTS = 30
pca_data, pca_model = pca_apply(data, COMPONENTS)


PATCH_SIZE = 11
train_labels_map, test_labels_map = checkerboard_split(labels, block_size=24, patch_size=PATCH_SIZE)
# --- 3. PATCH EXTRACTION ---

x_train_all, y_train_all = create_patches(pca_data, train_labels_map, PATCH_SIZE)
x_test_all, y_test_all = create_patches(pca_data, test_labels_map, PATCH_SIZE)

# Filtra o fundo (zeros) e ajusta as classes (0-indexed)
train_mask_1d = y_train_all > 0
x_training = x_train_all[train_mask_1d]

print(f"Total de patches para treino: {len(x_training)}")
y_train = y_train_all[train_mask_1d] - 1

test_mask_1d = y_test_all > 0
x_test = x_test_all[test_mask_1d]
y_test = y_test_all[test_mask_1d] - 1

print(f"Patches de Treino: {len(x_training)} | Patches de Teste: {len(x_test)}")

# Filter out background (0s) and shift to 0-indexed classes
train_mask_1d = y_train_all > 0
x_training = x_train_all[train_mask_1d]
y_train = y_train_all[train_mask_1d] - 1

test_mask_1d = y_test_all > 0
x_test = x_test_all[test_mask_1d]
y_test = y_test_all[test_mask_1d] - 1

print(f"Valid Training patches: {len(x_training)} | Valid Testing patches: {len(x_test)}")

# PYTORCH FORMATTING
x_training = np.transpose(x_training, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
y_training = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

train_ds = TensorDataset(torch.from_numpy(x_training), torch.from_numpy(y_training))
test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- 5. MODEL SETUP ---
num_classes = int(np.max(labels))
in_channel = pca_data.shape[-1]
model = CNN2D(channels=in_channel, classes=num_classes).to(using_gpu)

# Calculate class weights for imbalanced data
#class_counts = np.where(class_counts == 0, 1, class_counts) # Prevent division by zero
total_samples = len(y_train)

class_counts = np.bincount(y_train, minlength=num_classes)
class_counts = np.where(class_counts == 0, 1, class_counts)

#weights = np.log(total_samples / (class_counts + 1e-6))
#weights = weights / np.sum(weights) * num_classes # Normaliza
#
#class_weights = torch.tensor(weights, dtype=torch.float32).to(using_gpu)
#weights = total_samples / (num_classes * class_counts)
#class_weights = torch.tensor(weights, dtype=torch.float32).to(using_gpu)
#
## Loss and Optimizer
##criterion = nn.CrossEntropyLoss(weight=class_weights).to(using_gpu)
#criterion = nn.CrossEntropyLoss().to(using_gpu)
#optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

weights = 1.0 / (class_counts + 1)
weights = weights / weights.sum() * num_classes
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)

def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(using_gpu), yb.to(using_gpu)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    # Handle edge case where test loader might be empty based on split
    if total == 0: return 0.0, 0.0
    return loss_sum / total, correct / total

EPOCHS = 60
train_history = {'loss': [], 'val_loss': [], 'val_acc': []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(using_gpu), yb.to(using_gpu)

        xb = torch.flip(xb, dims=[np.random.choice([2, 3])])  # Flip aleatório
        if np.random.rand() > 0.5:
            xb = torch.rot90(xb, k=np.random.randint(1, 4), dims=[2, 3])  # Rotação aleatória


        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    val_loss, val_acc = evaluate(test_loader)

    train_history['loss'].append(train_loss)
    train_history['val_loss'].append(val_loss)
    train_history['val_acc'].append(val_acc)

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch:02d}/{EPOCHS} | lr={current_lr:.6f} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}')



results(pca_data, labels, train_labels_map, test_labels_map, model, PATCH_SIZE, using_gpu, train_history)

