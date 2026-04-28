import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from swin_transformer import SwinTransformer
from ANN_Models_Hyperspectral_Images.src.testing import (
    setup_device, load_data, pca_apply,
    random_split, extract_split_patches, results
)
using_gpu = setup_device()
DATA_PATH = r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/indianpinearray.npy"
LABEL_PATH = r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/IPgt.npy"
data, labels = load_data(DATA_PATH, LABEL_PATH)
COMPONENTS = 10
PATCH_SIZE = 32
pca_data, pca_model = pca_apply(data, COMPONENTS)
train_labels_map, test_labels_map = random_split(labels, test_size=0.5, random_state=42)

x_training, y_train = extract_split_patches(pca_data, train_labels_map, PATCH_SIZE)
x_test, y_test = extract_split_patches(pca_data, test_labels_map, PATCH_SIZE)

x_training = np.transpose(x_training, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
y_training = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

train_ds = TensorDataset(torch.from_numpy(x_training), torch.from_numpy(y_training))
test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

num_classes = int(np.max(labels))
model = SwinTransformer(in_channels=10, num_classes=num_classes, img_size=32).to(using_gpu)  # ← here

class_counts = np.bincount(y_train, minlength=num_classes)
class_counts = np.where(class_counts == 0, 1, class_counts)
weights = 1.0 / (class_counts + 1)
weights = weights / weights.sum() * num_classes
class_weights = torch.tensor(weights, dtype=torch.float32).to(using_gpu)
criterion = nn.CrossEntropyLoss(weight=class_weights).to(using_gpu)

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

            for i in range(len(yb)):
                label = yb[i].item()
                class_total[label] += 1
                if preds[i].item() ==  label:
                    class_correct[label] += 1

    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

    #OA
    oa = correct / total

    #AA
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

print(f"Train pixels: {(train_labels_map > 0).sum()}")
print(f"Test pixels:  {(test_labels_map > 0).sum()}")
print(f"Train patches: {len(x_training)}")
print(f"Test patches:  {len(x_test)}")
