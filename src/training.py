import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from CNN import CNN2D

from testing import random_split, extract_split_patches, setup_device, load_data, pca_apply, create_patches, results


using_gpu = setup_device()

DATA_PATH = r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/indianpinearray.npy"
LABEL_PATH = r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/IPgt.npy"

# 2. Pass the strings into your function to load the arrays
data, labels = load_data(DATA_PATH, LABEL_PATH)
print(f"Original Data Shape: {data.shape} | Labels Shape: {labels.shape}")

# Apply PCA
COMPONENTS = 20
pca_data, pca_model = pca_apply(data, COMPONENTS)


PATCH_SIZE = 11
train_labels_map, test_labels_map = random_split(labels, test_size=0.5, random_state=42)

#train_labels_map, test_labels_map = checkerboard_split(labels, block_size=24, patch_size=PATCH_SIZE)
#
x_train_all, y_train_all = create_patches(pca_data, train_labels_map, PATCH_SIZE)
x_test_all, y_test_all = create_patches(pca_data, test_labels_map, PATCH_SIZE)


#Extract only the valid patches and shift labels to 0-indexed
x_training, y_train = extract_split_patches(pca_data, train_labels_map, PATCH_SIZE)
x_test, y_test = extract_split_patches(pca_data, test_labels_map, PATCH_SIZE)

print(f"Valid Training patches: {len(x_training)} | Valid Testing patches: {len(x_test)}")
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


num_classes = int(np.max(labels))

in_channel = pca_data.shape[-1]
model = CNN2D(channels=in_channel, classes=num_classes).to(using_gpu)

# Calculate class weights for imbalanced data
#class_counts = np.where(class_counts == 0, 1, class_counts) # Prevent division by zero
total_samples = len(y_train)

class_counts = np.bincount(y_train, minlength=num_classes)
class_counts = np.where(class_counts == 0, 1, class_counts)


#class_weights = torch.tensor(weights, dtype=torch.float32).to(using_gpu)
## Loss and Optimizer
#criterion = nn.CrossEntropyLoss().to(using_gpu)

weights = 1.0 / (class_counts + 1)
weights = weights / weights.sum() * num_classes
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=15
)

def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0

    num_classes = model.classes if hasattr(model, 'classes') else int(np.max(labels))

    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(using_gpu), yb.to(using_gpu)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss_sum += loss.item() * xb.size(0)

            preds = logits.argmax(dim=1)

            correct += (preds == yb).sum().item()
            total += yb.size(0)

            # 🔹 Contagem por classe (para AA)
            for i in range(len(yb)):
                label = yb[i].item()
                class_total[label] += 1
                if preds[i].item() ==  label:
                    class_correct[label] += 1

    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

    #  OA
    oa = correct / total

    #  AA
    class_acc = np.divide(class_correct, class_total, out=np.zeros_like(class_correct), where=class_total != 0)
    aa = np.mean(class_acc)

    return loss_sum / total, oa, aa, class_acc

EPOCHS = 60
train_history = {'loss': [], 'val_loss': [], 'val_acc': [], 'val_oa' : [],'val_aa': []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(using_gpu), yb.to(using_gpu)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    val_loss, val_oa, val_aa, class_acc = evaluate(test_loader)


    train_history['loss'].append(train_loss)
    train_history['val_loss'].append(val_loss)
    train_history['val_oa'].append(val_oa)
    train_history['val_aa'].append(val_aa)

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch:02d}/{EPOCHS} | '
          f'lr={current_lr:.6f} | '
          f'train_loss={train_loss:.4f} | '
          f'val_loss={val_loss:.4f} | '
          f'OA={val_oa:.4f} | '
          f'AA={val_aa:.4f}')

results(pca_data, labels, train_labels_map, test_labels_map, model, PATCH_SIZE, using_gpu, train_history)

