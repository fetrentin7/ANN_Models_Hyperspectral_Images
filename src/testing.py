import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch

from sklearn.metrics import accuracy_score, recall_score

def setup_device():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    return device


def load_data(path, label):

    data = np.load(path)
    labels = np.load(label)

    return data, labels


def pca_apply(X, n_components):
    # reshapes the numpy array into a new one and the tuple is defining the array dimension (2)
    # -1 a placeholder to automatically calculate the required size for this dimension based on the total number of elements in the array and the size of the other dimension.
    new_x = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components, whiten=True)
    x_value = pca.fit_transform(new_x)
    new_X = np.reshape(x_value, (X.shape[0], X.shape[1], n_components))
    return new_X, pca

# creating the training patches
def create_patches(x, y, size):
    margin = size // 2
    padded_x = np.pad(x, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
    labels = []
    list = []
    for i in range(margin, padded_x.shape[0] - margin):
        for j in range(margin, padded_x.shape[1] - margin):
            patch = padded_x[i - margin:i + margin + 1, j - margin:j + margin + 1, :]
            list.append(patch)
            labels.append(y[i - margin, j - margin])

    return np.array(list), np.array(labels)


def random_split(labels, test_size=0.5, random_state=42):
    train_labels = np.zeros_like(labels)
    test_labels = np.zeros_like(labels)

    valid_indices = np.argwhere(labels > 0)
    valid_classes = labels[valid_indices[:, 0], valid_indices[:, 1]]

    train_idx, test_idx = train_test_split(
        valid_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=valid_classes
    )

    for r, c in train_idx:
        train_labels[r, c] = labels[r, c]
    for r, c in test_idx:
        test_labels[r, c] = labels[r, c]

    return train_labels, test_labels


def extract_split_patches(data_pca, label_map, patch_size):
    margin = patch_size // 2
    padded_data = np.pad(data_pca, ((margin, margin), (margin, margin), (0, 0)), mode='constant')

    patches = []
    y_labels = []
    valid_coords = np.argwhere(label_map > 0)

    for r, c in valid_coords:
        patch = padded_data[r: r + patch_size, c: c + patch_size, :]
        patches.append(patch)
        y_labels.append(label_map[r, c] - 1)  # Shift to 0-indexed for PyTorch

    return np.array(patches, dtype=np.float32), np.array(y_labels, dtype=np.int64)


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

           #  check if the block hasn't been completely swallowed by the margin, then split between test and training
           if safe_r_start < safe_r_end and safe_c_start < safe_c_end:
               if is_train:
                   train_labels[safe_r_start:safe_r_end, safe_c_start:safe_c_end] = \
                       labels[safe_r_start:safe_r_end, safe_c_start:safe_c_end]
               else:
                   test_labels[safe_r_start:safe_r_end, safe_c_start:safe_c_end] = \
                       labels[safe_r_start:safe_r_end, safe_c_start:safe_c_end]

    return train_labels, test_labels

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

#def disjoint_split(X, y, split_ratio=0.5):
#    h, w, _ = X.shape
#    split_col = int(w * split_ratio)
#
#    # esquerda = treino
#    X_train = X[:, :split_col, :]
#    y_train = y[:, :split_col]
#
#    # direita = teste
#    X_test = X[:, split_col:, :]
#    y_test = y[:, split_col:]
#
#    return X_train, X_test, y_train, y_test


def results(data_pca, labels, train_labels, test_labels, model, patch_size, using_gpu, train_history=None):
    model.eval()
    margin = patch_size // 2
    padded = np.pad(data_pca, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
    patches = []

    # Extract all patches across the entire map
    for i in range(margin, padded.shape[0] - margin):
        for j in range(margin, padded.shape[1] - margin):
            px = padded[i - margin: i+margin+1, j - margin: j+margin+1, :]
            patches.append(px)

    patches = np.array(patches, dtype=np.float32)

    # Convert to PyTorch format (N, C, H, W)
    patches_t = torch.from_numpy(np.transpose(patches, (0, 3, 1, 2)))
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

    # Reshape predictions back to the image dimensions
    predicted_labels = np.concatenate(preds_all, axis=0).reshape(labels.shape)
    predicted_labels_shifted = predicted_labels + 1  # Shift back to 1-indexed classes

    # --- CALCULATE SEPARATE TRAIN & TEST ACCURACIES ---
    train_mask = train_labels != 0
    test_mask = test_labels != 0

    correct_train = (predicted_labels_shifted[train_mask] == train_labels[train_mask]).sum()
    total_train = train_mask.sum()
    train_acc = correct_train / total_train if total_train > 0 else 0

    correct_test = (predicted_labels_shifted[test_mask] == test_labels[test_mask]).sum()
    total_test = test_mask.sum()
    test_acc = correct_test / total_test if total_test > 0 else 0

    print(f"Final Map Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Final Map Testing Accuracy:  {test_acc * 100:.2f}%")

    # --- CREATE SEPARATE ERROR MAPS ---

    train_error_map = np.zeros_like(labels)
    train_misclassified = train_mask & (predicted_labels_shifted != train_labels)
    train_error_map[train_misclassified] = 1

    test_error_map = np.zeros_like(labels)
    test_misclassified = test_mask & (predicted_labels_shifted != test_labels)
    test_error_map[test_misclassified] = 1

    predicted_labels_masked = predicted_labels_shifted * (labels != 0)

    correct_test = (predicted_labels_shifted[test_mask] == test_labels[test_mask]).sum()
    total_test = test_mask.sum()
    test_acc = correct_test / total_test if total_test > 0 else 0


    y_test_true = test_labels[test_mask]
    y_test_pred = predicted_labels_shifted[test_mask]

    oa_test = accuracy_score(y_test_true, y_test_pred)
    aa_test = recall_score(y_test_true, y_test_pred, average='macro', zero_division=0)

    plt.figure(figsize=(8, 6))
    metrics = ['Overall Acc (OA)', 'Average Acc (AA)']
    values = [oa_test * 100, aa_test * 100]
    color = ['#2ca02c', '#1f77b4']

    bars = plt.bar(metrics, values, color=color, width=0.5)
    plt.ylim(0, 105)
    plt.ylabel('Porcentagem (%)')
    plt.title('Métricas de Desempenho do Modelo (Teste)')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}%',
                 ha='center', va='bottom', fontweight='bold')

    hardware_text = (
        f"Eficiência de Hardware:\n"
        f"• Memória de Vídeo: {memory_used_mb:.2f} MB\n"
        f"• Tempo Inferência: {inference_time:.4f} s"
    )

    if train_history is not None and len(train_history['loss']) > 0:
        epochs = range(1, len(train_history['loss']) + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_history['loss'], label='Train Loss', marker='.')
        plt.plot(epochs, train_history['val_loss'], label='Val Loss', marker='.')
        plt.title('Learning Curve: Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_history['val_oa'], label='Val OA', marker='.')
        plt.plot(epochs, train_history['val_aa'], label='Val AA', marker='.')
        plt.title('Learning Curve: Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # Plot the Inference and Error Maps
    from matplotlib.colors import ListedColormap
    error_cmap = ListedColormap(['white', 'red'])

    plt.figure(figsize=(15, 10))

    # Top Row: Ground Truth & Overall Prediction
    plt.subplot(2, 3, 1)
    plt.title('Ground Truth (Full)')
    plt.imshow(labels, cmap='jet', interpolation='nearest')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Predicted (Full Inference)")
    plt.imshow(predicted_labels_shifted, cmap='jet', interpolation='nearest')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Predicted (Masked)')
    plt.imshow(predicted_labels_masked, cmap='jet', interpolation='nearest')
    plt.axis('off')

    # Bottom Row: The Train/Test Split and Specific Error Maps
    plt.subplot(2, 3, 4)
    # Combine train (blue) and test (green) masks to visualize the checkerboard
    split_vis = np.zeros_like(labels)
    split_vis[train_mask] = 1  # Train
    split_vis[test_mask] = 2   # Test
    split_cmap = ListedColormap(['black', 'blue', 'green'])
    plt.title('Split (Blue=Train, Green=Test)')
    plt.imshow(split_vis, cmap=split_cmap, interpolation='nearest')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title(f'Train Error Map\nAcc: {train_acc*100:.1f}%')
    plt.imshow(train_error_map, cmap=error_cmap, interpolation='nearest')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title(f'Test Error Map\nAcc: {test_acc*100:.1f}%')
    plt.imshow(test_error_map, cmap=error_cmap, interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

