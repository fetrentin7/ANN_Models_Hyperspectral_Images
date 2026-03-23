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
    # -1 a placehol der to automatically calculate the required size for this dimension based on the total number of elements in the array and the size of the other dimension.
    new_x = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components, whiten=True)
    x_value = pca.fit_transform(new_x)
    new_X = np.reshape(x_value, (X.shape[0], X.shape[1], n_components))
    return new_X, pca

#apply pca

pca_data, pca_model = pca_apply(DATA_PATH, 150)
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
            patch = padded_x[i-margin:i+margin+1, j-margin:j+margin+1, :]
            list.append(patch)
            labels.append(y[i - margin, j - margin])

    return np.array(list), np.array(labels)


""""def half_and_half_split(labels, patch_size=15):
    Splits the map cleanly in half (Left = Train, Right = Test), with a dead zone in the middle to prevent overlapping patches.
    
    train_labels = np.zeros_like(labels)
    test_labels = np.zeros_like(labels)

    rows, cols = labels.shape
    midpoint = cols // 2
    buffer = patch_size  # The dead zone trench

    # Left side for training
    train_labels[:, :midpoint - buffer] = labels[:, :midpoint - buffer]

    # Right side for testing
    test_labels[:, midpoint + buffer:] = labels[:, midpoint + buffer:]

    return train_labels, test_labels"""


def checkerboard_split(labels, block_size, patch_size):

    train_labels = np.zeros_like(labels)
    test_labels = np.zeros_like(labels)
    margin = patch_size // 2  # The dead zone size (7 pixels)

    rows, cols = labels.shape

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            # Determine grid position
            grid_row = i // block_size
            grid_col = j // block_size

            # If row + col is even, it's a Train block. Otherwise, Test block.
            is_train = (grid_row + grid_col) % 2 == 0

            r_end = min(i + block_size, rows)
            c_end = min(j + block_size, cols)

            # Apply the dead zone margin to the inside of the block
            safe_r_start = i + margin
            safe_r_end = r_end - margin
            safe_c_start = j + margin
            safe_c_end = c_end - margin

            #  check if the block hasn't been completely swallowed by the margin
            if safe_r_start < safe_r_end and safe_c_start < safe_c_end:
                if is_train:
                    train_labels[safe_r_start:safe_r_end, safe_c_start:safe_c_end] = \
                        labels[safe_r_start:safe_r_end, safe_c_start:safe_c_end]
                else:
                    test_labels[safe_r_start:safe_r_end, safe_c_start:safe_c_end] = \
                        labels[safe_r_start:safe_r_end, safe_c_start:safe_c_end]

    return train_labels, test_labels

PATCH_SIZE = 7

# Create the checkerboard map
train_labels_map, test_labels_map = checkerboard_split(labels, block_size=15, patch_size=PATCH_SIZE)

# Extract patches separately
x_train_all, y_train_all = create_patches(pca_data, train_labels_map, PATCH_SIZE)
x_test_all, y_test_all = create_patches(pca_data, test_labels_map, PATCH_SIZE)

# Filter out the background (0s) and shift to 0-indexed classes
train_mask = y_train_all > 0
x_training = x_train_all[train_mask]
y_train = y_train_all[train_mask] - 1

test_mask = y_test_all > 0
x_test = x_test_all[test_mask]
y_test = y_test_all[test_mask] - 1

train_classes, train_counts = np.unique(y_train, return_counts=True)
test_classes, test_counts = np.unique(y_test, return_counts=True)
print(f"Train Classes: {train_classes} \nCounts: {train_counts}")
print(f"Test Classes: {test_classes} \nCounts: {test_counts}")


print(f"Training samples: {len(x_training)} | Testing samples: {len(x_test)}")


#converting to pytorch tensors --> switch to channels first (N,C,H,W)

x_training = np.transpose(x_training, (0,3,1,2)).astype(np.float32)
x_test = np.transpose(x_test, (0,3,1,2)).astype(np.float32)
y_training = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

train_ds = TensorDataset(torch.from_numpy(x_training), torch.from_numpy(y_training))
test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

num_classes = int(np.max(labels))
in_channel = pca_data.shape[-1]
model = CNN2D(channels=in_channel, classes=num_classes).to(using_gpu)

# --- CALCULATE CLASS WEIGHTS HERE ---
# Now num_classes and y_train both exist!
class_counts = np.bincount(y_train, minlength=num_classes)
class_counts = np.where(class_counts == 0, 1, class_counts)

total_samples = len(y_train)
weights = total_samples / (num_classes * class_counts)

class_weights = torch.tensor(weights, dtype=torch.float32).to(using_gpu)
# ------------------------------------

# Pass the weights to the teacher (Loss Function)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


EPOCHS = 120

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
    patches = []

    for i in range(margin, padded.shape[0] - margin):
        for j in range(margin, padded.shape[1] - margin):
            px = padded[i - margin: i+margin+1, j - margin: j+margin+1,:]
            patches.append(px)

    patches = np.array(patches, dtype=np.float32)

    #converting to pytorch
    patches_t = torch.from_numpy(np.transpose(patches, (0,3,1,2)))
    preds_all = []

    start_time = time.time()

    with torch.no_grad():
        for start in range(0, patches_t.shape[0], 64):
            batch = patches_t[start:start + 64].to(using_gpu)
            logits = model(batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            preds_all.append(preds)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Total Inference Time: {inference_time:.4f} seconds")

    memory_used_mb = torch.cuda.max_memory_allocated(using_gpu) / (1024 * 1024)
    print(f"Peak GPU Memory Used: {memory_used_mb:.2f} MB")

    predicted_labels = np.concatenate(preds_all, axis=0).reshape(labels.shape)
    predicted_labels_masked = predicted_labels * (labels != 0)

    predicted_labels_shifted = predicted_labels + 1
    predicted_labels_masked = predicted_labels_shifted * (labels != 0)

    #  Create an Error Map (1 for error, 0 for correct or background) ---
    error_map = np.zeros_like(labels)
    # Check where predictions don't match labels, ignoring the background
    misclassified = (labels != 0) & (predicted_labels_shifted != labels)
    error_map[misclassified] = 1

    # --- 1. Plot the Accuracy Graph ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_history['val_acc'], label='Overall Accuracy (val_acc)', color='blue')
    plt.title('Overall Accuracy x Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # --- 2. Plot the Inference Maps ---
    # Changed from (1,3) to (2,2) to accommodate the new Error Map
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.title('Ground Truth')
    plt.imshow(labels, cmap='jet')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Predicted (Full Inference)")
    plt.imshow(predicted_labels_shifted, cmap='jet')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title('Predicted (Masked / No Background)')
    plt.imshow(predicted_labels_masked, cmap='jet')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Error Map (Red = Misclassified)')
    # Using a custom colormap for the error map: white for correct/background, red for errors
    from matplotlib.colors import ListedColormap
    error_cmap = ListedColormap(['white', 'red'])
    plt.imshow(error_map, cmap=error_cmap)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

results(pca_data, labels, model, PATCH_SIZE)

