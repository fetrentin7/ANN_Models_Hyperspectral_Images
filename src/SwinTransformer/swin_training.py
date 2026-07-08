import numpy as np
import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ANN_Models_Hyperspectral_Images.src.load_dataset import choose_dataset

from torch.utils.data import TensorDataset, DataLoader

from swin_transformer import SwinTransformer
from ANN_Models_Hyperspectral_Images.src.testing import (
    setup_device, load_data, pca_apply,
    random_split, extract_split_patches, results
)

best_score = 0.0
best_model_state = None
best_epoch = 0
using_gpu = setup_device()
#DATA_PATH = r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/indianpinearray.npy"
#LABEL_PATH = r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/IPgt.npy"


DATASET, DATA_PATH, LABEL_PATH, CLASS_NAMES = choose_dataset()
data, labels = load_data(DATA_PATH, LABEL_PATH)
COMPONENTS = 50
PATCH_SIZE = 32 #ANTES TAVA 64 NO KSC

pca_data, pca_model = pca_apply(data, COMPONENTS)
train_labels_map, test_labels_map = random_split(labels, test_size=0.5, random_state=42)

x_training, y_train = extract_split_patches(pca_data, train_labels_map, PATCH_SIZE)
x_test, y_test = extract_split_patches(pca_data, test_labels_map, PATCH_SIZE)

x_training = np.transpose(x_training, (0, 3, 1, 2)).astype(np.float32)
print(f"Valid Training patches: {len(x_training)} ...")

x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
y_training = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

train_ds = TensorDataset(torch.from_numpy(x_training), torch.from_numpy(y_training))
test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

BATCH_SIZE = 2048
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

num_classes = int(np.max(labels))
model = SwinTransformer(in_channels=COMPONENTS, num_classes=num_classes, img_size=PATCH_SIZE).to(using_gpu)

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
EPOCHS = 200
MIN = 0.01
PATIENCE = 50
train_history = {'loss': [], 'val_loss': [], 'val_oa': [], 'val_aa': []}

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

    score = (val_oa + val_aa) / 2
    if score > (best_score + MIN):
        best_score = score
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
        epochs_without_improvement = 0

    else:
        epochs_without_improvement +=1

    train_history['loss'].append(train_loss)
    train_history['val_loss'].append(val_loss)
    train_history['val_oa'].append(val_oa)
    train_history['val_aa'].append(val_aa)

    scheduler.step(val_loss)
    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}. "
              f"Best epoch: {best_epoch}, best score: {best_score:.4f}")
        break

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch:02d}/{EPOCHS} | '
          f'lr={current_lr:.6f} | '
          f'train_loss={train_loss:.4f} | '
          f'val_loss={val_loss:.4f} | '
          f'OA={val_oa:.4f} | '
          f'AA={val_aa:.4f}')

    if epoch % 10 == 0 and using_gpu.type == 'cuda':
        current = torch.cuda.memory_allocated(using_gpu) / (1024 * 1024)
        peak = torch.cuda.max_memory_allocated(using_gpu) / (1024 * 1024)
        print(f"  [GPU] current={current:.1f} MB | peak={peak:.1f} MB")

# After the loop ends — capture final training peak
training_peak_mb = torch.cuda.max_memory_allocated(using_gpu) / (1024 * 1024)
print(f"Peak GPU Memory During Training: {training_peak_mb:.2f} MB")

print(f"Train pixels: {(train_labels_map > 0).sum()}")
print(f"Test pixels:  {(test_labels_map > 0).sum()}")
print(f"Train patches: {len(x_training)}")
print(f"Test patches:  {len(x_test)}")


results(pca_data, labels, train_labels_map, test_labels_map,
        model, PATCH_SIZE, using_gpu, train_history,
        class_names=CLASS_NAMES)