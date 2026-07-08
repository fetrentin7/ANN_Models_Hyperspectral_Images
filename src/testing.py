import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, average_precision_score,precision_recall_curve
import xai
import pandas as pd
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from matplotlib.colors import ListedColormap

IP_CLASSES = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
              "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
              "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
              "Soybean-clean", "Wheat", "Woods",
              "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]

PU_CLASSES = ["Asphalt", "Meadows", "Gravel", "Trees",
              "Painted-metal-sheets", "Bare-Soil", "Bitumen",
              "Self-Blocking-Bricks", "Shadows"]
KSC_CLASSES = ["Scrub",
               "Willow swamp",
               "Cabbage palm hammock",
               "Cabbage palm/oak hammock",
               "Slash pine",
               "Oak/broadleaf hammock",
               "Hardwood swamp",
               "Graminoid marsh",
               "Spartina marsh",
               "Cattail marsh",
               "Salt marsh",
               "Mud flats",
               "Water"]
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
#def create_patches(x, y, size):
#    margin = size // 2
#    padded_x = np.pad(x, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
#    labels = []
#    list = []
#    for i in range(margin, padded_x.shape[0] - margin):
#        for j in range(margin, padded_x.shape[1] - margin):
#            patch = padded_x[i - margin:i + margin + 1, j - margin:j + margin + 1, :]
#            list.append(patch)
#            labels.append(y[i - margin, j - margin])
#
#    return np.array(list), np.array(labels)
#

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


#def checkerboard_split(labels, block_size, patch_size):
#    train_labels = np.zeros_like(labels)
#    test_labels = np.zeros_like(labels)
#    margin = patch_size // 2  # The dead zone size (7 pixels)
#    rows, cols = labels.shape
#    for i in range(0, rows, block_size):
#       for j in range(0, cols, block_size):
#           # Determine grid position
#           grid_row = i // block_size
#           grid_col = j // block_size
#
#           # If row + col is even, it's a Train block. Otherwise, Test block.
#           is_train = (grid_row + grid_col) % 2 == 0
#           r_end = min(i + block_size, rows)
#           c_end = min(j + block_size, cols)
#
#           # Apply the dead zone margin to the inside of the block
#           safe_r_start = i + margin
#           safe_r_end = r_end - margin
#           safe_c_start = j + margin
#           safe_c_end = c_end - margin
#
#           #  check if the block hasn't been completely swallowed by the margin, then split between test and training
#           if safe_r_start < safe_r_end and safe_c_start < safe_c_end:
#               if is_train:
#                   train_labels[safe_r_start:safe_r_end, safe_c_start:safe_c_end] = \
#                       labels[safe_r_start:safe_r_end, safe_c_start:safe_c_end]
#               else:
#                   test_labels[safe_r_start:safe_r_end, safe_c_start:safe_c_end] = \
#                       labels[safe_r_start:safe_r_end, safe_c_start:safe_c_end]
#
#    return train_labels, test_labels

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

def disjoint_split(labels, split_ratio=0.5, buffer=None):
    train_labels = np.zeros_like(labels)
    test_labels = np.zeros_like(labels)
    h, w = labels.shape
    split_col = int(w * split_ratio)
    if buffer is None:
        buffer = 0
    train_labels[:, :split_col - buffer] = labels[:, :split_col - buffer]
    test_labels[:, split_col + buffer:] = labels[:, split_col + buffer:]
    return train_labels, test_labels

def results(data_pca, labels, train_labels, test_labels, model, patch_size,
            using_gpu, train_history=None, class_names=None):
    model.eval()
    margin = patch_size // 2
    padded = np.pad(data_pca, ((margin, margin), (margin, margin), (0, 0)), mode='constant')

    #  Extract all patches across the entire map
    H, W = labels.shape
    total_pixels = H * W
    CHUNK_SIZE = 256

    print(f"DEBUG: processing {total_pixels} patches in chunks of {CHUNK_SIZE}")

    if using_gpu.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(using_gpu)

    preds_all = []
    proba_all = []
    start_time = time.time()

    # Build coordinate list once (small — just integers)
    coords = [(r, c) for r in range(H) for c in range(W)]

    with torch.no_grad():
        for chunk_start in range(0, total_pixels, CHUNK_SIZE):
            chunk_coords = coords[chunk_start:chunk_start + CHUNK_SIZE]

            # Build only this chunk's patches
            chunk_patches = np.empty((len(chunk_coords), patch_size, patch_size, padded.shape[2]),
                                     dtype=np.float32)
            for i, (r, c) in enumerate(chunk_coords):
                chunk_patches[i] = padded[r: r + patch_size, c: c + patch_size, :]

            # Convert to torch (N, C, H, W) and send to GPU
            chunk_t = torch.from_numpy(np.transpose(chunk_patches, (0, 3, 1, 2))).to(using_gpu)

            logits = model(chunk_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            preds = proba.argmax(axis=1)
            preds_all.append(preds)
            proba_all.append(proba)

            # Free chunk memory
            del chunk_patches, chunk_t

    inference_time = time.time() - start_time
    print(f"Total Inference Time: {inference_time:.4f} seconds")

    inference_per_sample = inference_time / total_pixels
    print(f"Inference per sample: {inference_per_sample * 1000:.4f} ms")

    if using_gpu.type == 'cuda':
        memory_used_mb = torch.cuda.max_memory_allocated(using_gpu) / (1024 * 1024)
        print(f"Peak GPU Memory Used: {memory_used_mb:.2f} MB")
    else:
        memory_used_mb = 0.0
        print("Running on CPU — GPU memory metric skipped.")

    #Reshape predictions to image dims
    predicted_labels = np.concatenate(preds_all, axis=0).reshape(labels.shape)
    predicted_proba = np.concatenate(proba_all, axis=0)  # (H*W, num_classes)
    predicted_labels_shifted = predicted_labels + 1  # back to 1-indexed

    #Train / Test accuracies on the map
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

    #Error maps
    train_error_map = np.zeros_like(labels)
    train_error_map[train_mask & (predicted_labels_shifted != train_labels)] = 1

    test_error_map = np.zeros_like(labels)
    test_error_map[test_mask & (predicted_labels_shifted != test_labels)] = 1

    predicted_labels_masked = predicted_labels_shifted * (labels != 0)

    # repare arrays for metrics
    y_test_true = test_labels[test_mask]
    y_test_pred = predicted_labels_shifted[test_mask]

    oa_test = accuracy_score(y_test_true, y_test_pred)
    aa_test = recall_score(y_test_true, y_test_pred, average='macro', zero_division=0)

    y_test_true_0idx = (y_test_true - 1).astype(int)
    y_test_pred_0idx = (y_test_pred - 1).astype(int)

    # Probabilities aligned with test_mask
    y_test_proba = predicted_proba.reshape(labels.shape[0], labels.shape[1], -1)
    y_test_proba = y_test_proba[test_mask]  # (N_test, num_classes)

    num_classes = predicted_proba.shape[1]
    if class_names is None:
        class_names = IP_CLASSES[:num_classes]

    # PLOT 1: OA / AA bar chart with hardware efficiency caption
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics_labels = ['Overall Acc (OA)', 'Average Acc (AA)']
    values = [oa_test * 100, aa_test * 100]
    color = ['#2ca02c', '#1f77b4']

    bars = ax.bar(metrics_labels, values, color=color, width=0.5)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Porcentagem (%)')
    ax.set_title('Métricas de Desempenho do Modelo (Teste)')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}%',
                ha='center', va='bottom', fontweight='bold')

    hardware_text = (
        f"Eficiência de Hardware:\n"
        f"• Memória de Vídeo: {memory_used_mb:.2f} MB\n"
        f"• Tempo Inferência: {inference_time:.4f} s"
    )
    ax.text(0.98, 0.98, hardware_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.show()  # FIX 5


    # PLOT 2: Learning curves

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
        plt.xlabel('Epoch');
        plt.ylabel('Accuracy')
        plt.legend();
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    # PLOT 3: Inference and Error Maps

    error_cmap = ListedColormap(['white', 'red'])
    plt.figure(figsize=(15, 10))

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

    plt.subplot(2, 3, 4)
    split_vis = np.zeros_like(labels)
    split_vis[train_mask] = 1
    split_vis[test_mask] = 2
    split_cmap = ListedColormap(['black', 'blue', 'green'])
    plt.title('Split (Blue=Train, Green=Test)')
    plt.imshow(split_vis, cmap=split_cmap, interpolation='nearest')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title(f'Train Error Map\nAcc: {train_acc * 100:.1f}%')
    plt.imshow(train_error_map, cmap=error_cmap, interpolation='nearest')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title(f'Test Error Map\nAcc: {test_acc * 100:.1f}%')
    plt.imshow(test_error_map, cmap=error_cmap, interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # ===========================================================
    # ============= XAI BLOCK (consistent standard) =============
    # ===========================================================
    print("\n" + "=" * 50)
    print("  XAI EVALUATION PLOTS")
    print("=" * 50)

    # --- XAI 1) Class imbalance of the labeled dataset ---
    valid_labels = labels[labels > 0]
    df_classes = pd.DataFrame({
        'class': [class_names[int(c) - 1] for c in valid_labels]
    })
    try:
        xai.imbalance_plot(df_classes, "class", categorical_cols=["class"])
        plt.title("Class Imbalance")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"xai.imbalance_plot failed: {e}")

    # --- ROC per class (sklearn — xai is binary-only) ---
    y_true_bin = label_binarize(y_test_true_0idx, classes=list(range(num_classes)))

    auc_pr = average_precision_score(y_true_bin, y_test_proba, average='macro')
    print(f"AUC-PR (macro): {auc_pr:.4f}")


    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        if y_true_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_test_proba[:, i])
        plt.plot(fpr, tpr, lw=1.5,
                 label=f'{class_names[i]} (AUC={auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate')
    plt.title('Per-class ROC')
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    plt.grid(alpha=0.3);
    plt.tight_layout()
    plt.show()

    # --- Curva Precisão-Revocação (AUC-PR) por classe ---

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        if y_true_bin[:, i].sum() == 0:
            continue
        precisao, revocacao, _ = precision_recall_curve(y_true_bin[:, i], y_test_proba[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_test_proba[:, i])
        plt.plot(revocacao, precisao, lw=1.5,
                 label=f'{class_names[i]} (AP={ap:.3f})')
    plt.xlabel('Revocação')
    plt.ylabel('Precisão')
    plt.title(f'Curva Precisão-Revocação por classe (AUC-PR macro={auc_pr:.3f})')
    plt.legend(loc='lower left', fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Confusion matrix (seaborn — xai is binary-only) ---
    class_ids = np.unique(np.concatenate([y_test_true, y_test_pred]))
    cm = confusion_matrix(y_test_true, y_test_pred, labels=class_ids)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_ids, yticklabels=class_ids)
    plt.xlabel('Predicted class');
    plt.ylabel('True class')
    plt.title(f'Confusion Matrix (Test)  OA={oa_test * 100:.2f}%  AA={aa_test * 100:.2f}%')
    plt.tight_layout()
    plt.show()

    # --- Normalized confusion matrix ---
    cm_norm = confusion_matrix(y_test_true, y_test_pred, labels=class_ids, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_ids, yticklabels=class_ids, vmin=0, vmax=1)
    plt.xlabel('Predicted class');
    plt.ylabel('True class')
    plt.title('Confusion Matrix (normalized by true class)')
    plt.tight_layout()
    plt.show()

    # --- Classification report ---
    print("\nClassification Report:")
    print(classification_report(y_test_true_0idx, y_test_pred_0idx,
                                target_names=class_names,
                                zero_division=0, digits=3))
