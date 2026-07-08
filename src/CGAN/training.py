import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import copy
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ANN_Models_Hyperspectral_Images.src.load_dataset import choose_dataset
from model.generator import Generator
from model.discriminator import Discriminator
from ANN_Models_Hyperspectral_Images.src.testing import (
    setup_device, load_data, pca_apply,
    random_split, extract_split_patches, results
)
using_gpu = setup_device()
#DATA_PATH = r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/indianpinearray.npy"
#LABEL_PATH = r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/IPgt.npy"


DATASET, DATA_PATH, LABEL_PATH, CLASS_NAMES = choose_dataset()
data, labels = load_data(DATA_PATH, LABEL_PATH)
best_score = 0.0
best_D_state = None
best_epoch = 0
best_oa = 0.0      # also track for the print at the end
best_aa = 0.0


COMPONENTS = 50
PATCH_SIZE = 32
pca_data, pca_model = pca_apply(data, COMPONENTS)
train_labels_map, test_labels_map = random_split(labels, test_size=0.5, random_state=42)

x_training, y_train = extract_split_patches(pca_data, train_labels_map, PATCH_SIZE)
x_test, y_test      = extract_split_patches(pca_data, test_labels_map,  PATCH_SIZE)

x_training = np.transpose(x_training, (0, 3, 1, 2)).astype(np.float32)
x_test     = np.transpose(x_test,     (0, 3, 1, 2)).astype(np.float32)
y_training = y_train.astype(np.int64)
y_test     = y_test.astype(np.int64)

train_ds = TensorDataset(torch.from_numpy(x_training), torch.from_numpy(y_training))
test_ds  = TensorDataset(torch.from_numpy(x_test),     torch.from_numpy(y_test))

BATCH_SIZE = 1024
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

#CGAN
num_classes = int(np.max(labels))
noise_dim   = 100
weight        = 1.0

G = Generator(noise_dim=noise_dim, num_classes=num_classes, patch_size=PATCH_SIZE, in_channels=COMPONENTS).to(using_gpu)
D = Discriminator(in_channels=COMPONENTS, patch_size=PATCH_SIZE, num_classes=num_classes).to(using_gpu)

opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

class_counts = np.bincount(y_train, minlength=num_classes)
class_counts = np.where(class_counts == 0, 1, class_counts)
weights = 1.0 / (class_counts + 1)
weights = weights / weights.sum() * num_classes
class_weights = torch.tensor(weights, dtype=torch.float32).to(using_gpu)
adversarial_loss    = nn.BCELoss()
classification_loss = nn.CrossEntropyLoss(weight=class_weights)

from torchinfo import summary

# Discriminador (recebe imagem)
print("=== DISCRIMINADOR ===")
summary(D, input_size=(1, COMPONENTS, PATCH_SIZE, PATCH_SIZE))

# Gerador (recebe ruído + rótulo)
print("=== GERADOR ===")
z = torch.randn(1, noise_dim).to(using_gpu)
lbl = torch.zeros(1, dtype=torch.long).to(using_gpu)
summary(G, input_data=(z, lbl))


sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_G, mode='max', factor=0.5, patience=15, min_lr=1e-6)
sched_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_D, mode='max', factor=0.5, patience=15, min_lr=1e-6)

class DiscriminatorClassifier(nn.Module):
    def __init__(self, discriminator):
        super().__init__()
        self.D = discriminator

    def forward(self, x):
        _, class_pred = self.D(x)
        return class_pred

# Evaluate
def evaluate(loader):
    D.eval()
    correct, total, loss_sum = 0, 0, 0.0
    class_correct = np.zeros(num_classes)
    class_total   = np.zeros(num_classes)

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(using_gpu), yb.to(using_gpu)

            _, class_pred = D(xb)
            loss = classification_loss(class_pred, yb)
            loss_sum += loss.item() * xb.size(0)

            preds = class_pred.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)

            for i in range(len(yb)):
                label = yb[i].item()
                class_total[label] += 1
                if preds[i].item() == label:
                    class_correct[label] += 1

    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

    oa = correct / total
    class_acc = np.divide(class_correct, class_total,
                          out=np.zeros_like(class_correct),
                          where=class_total != 0)
    aa = np.mean(class_acc)
    return loss_sum / total, oa, aa, class_acc

#training
EPOCHS = 700
train_history = {'loss': [], 'g_loss': [], 'val_loss': [], 'val_oa': [], 'val_aa': []}
best_score = 0.0
PATIENCE= 150
MIN = 0.001
best_model_state = None
for epoch in range(1, EPOCHS + 1):
    G.train(); D.train()
    d_running, g_running = 0.0, 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(using_gpu), yb.to(using_gpu)
        B = xb.size(0)

        real = torch.ones(B, 1).to(using_gpu)
        fake = torch.zeros(B, 1).to(using_gpu)

        #Train Discriminator
        opt_D.zero_grad()
        real_validity, real_class = D(xb)
        d_real = adversarial_loss(real_validity, real)
        d_cls  = classification_loss(real_class, yb)

        z = torch.randn(B, noise_dim).to(using_gpu)
        fake_patches = G(z, yb, xb).detach()
        fake_validity, _ = D(fake_patches)
        d_fake = adversarial_loss(fake_validity, fake)

        d_loss = d_real + d_fake + weight * d_cls
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
        opt_D.step()

        # Train Generator
        opt_G.zero_grad()
        z = torch.randn(B, noise_dim).to(using_gpu)
        gen_patches = G(z, yb, xb)
        fake_validity, fake_class = D(gen_patches)

        g_loss = adversarial_loss(fake_validity, real) + weight * classification_loss(fake_class, yb)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        opt_G.step()

        d_running += d_loss.item()
        g_running += g_loss.item()

    #Evaluation
    val_loss, val_oa, val_aa, class_acc = evaluate(test_loader)

    score = (val_oa + val_aa) / 2
    sched_G.step(score)
    sched_D.step(score)
    if score > (best_score + MIN):
        best_score = score
        best_D_state = {k: v.clone() for k, v in D.state_dict().items()}
        best_epoch = epoch
        best_oa = val_oa
        best_aa = val_aa
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    train_history['loss'].append(d_running / len(train_loader))
    train_history['g_loss'].append(g_running / len(train_loader))
    train_history['val_loss'].append(val_loss)
    train_history['val_oa'].append(val_oa)
    train_history['val_aa'].append(val_aa)

    current_lr = opt_D.param_groups[0]['lr']  # same as opt_G


    print(f'Epoch {epoch:02d}/{EPOCHS} | '
          f'lr={current_lr:.6f} | '
          f'D_loss={d_running / len(train_loader):.4f} | '
          f'G_loss={g_running / len(train_loader):.4f} | '
          f'val_loss={val_loss:.4f} | '
          f'OA={val_oa:.4f} | '
          f'AA={val_aa:.4f}')
    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}. "
              f"Best epoch: {best_epoch}, best score: {best_score:.4f}")
        break

    # GPU memory print every 10 epochs
    if epoch % 10 == 0 and using_gpu.type == 'cuda':
        current = torch.cuda.memory_allocated(using_gpu) / (1024 * 1024)
        peak = torch.cuda.max_memory_allocated(using_gpu) / (1024 * 1024)
        print(f"  [GPU] current={current:.1f} MB | peak={peak:.1f} MB")

# After training loop ends
training_peak_mb = torch.cuda.max_memory_allocated(using_gpu) / (1024 * 1024)
print(f"\nPeak GPU Memory During Training: {training_peak_mb:.2f} MB")

print(f"Train pixels: {(train_labels_map > 0).sum()}")
print(f"Test pixels:  {(test_labels_map > 0).sum()}")
print(f"Train patches: {len(x_training)}")
print(f"Test patches:  {len(x_test)}")

#  Load best Discriminator and run results
print(f"\nBest epoch: {best_epoch} | OA={best_oa:.4f} | AA={best_aa:.4f} | score={best_score:.4f}")
D.load_state_dict(best_D_state)

# Wrap D so results() gets only class logits, not the (validity, class) tuple
D_clf = DiscriminatorClassifier(D)

results(pca_data, labels, train_labels_map, test_labels_map,
        D_clf, PATCH_SIZE, using_gpu, train_history,
        class_names=CLASS_NAMES)