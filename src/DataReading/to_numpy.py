import scipy.io as sio
import numpy as np
import os


def convert_mat_to_npy(mat_data_path, mat_label_path, out_data_path, out_label_path):
    # Load .mat files
    data_mat = sio.loadmat(mat_data_path)
    label_mat = sio.loadmat(mat_label_path)

    # Find the actual array key (ignore __header__, __version__, __globals__)
    data_key = [k for k in data_mat.keys() if not k.startswith('__')][0]
    label_key = [k for k in label_mat.keys() if not k.startswith('__')][0]

    data = data_mat[data_key]
    labels = label_mat[label_key]

    print(f"  Data key:   '{data_key}' → shape {data.shape}, dtype {data.dtype}")
    print(f"  Label key:  '{label_key}' → shape {labels.shape}, dtype {labels.dtype}")
    print(f"  Label values: {np.unique(labels)}")

    # Save as .npy
    np.save(out_data_path, data)
    np.save(out_label_path, labels)
    print(f"  Saved → {out_data_path}")
    print(f"  Saved → {out_label_path}\n")


if __name__ == "__main__":
    # Adjust this to wherever your src folder is
    BASE = r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets"

    # ---- Pavia University ----
   #print("Converting Pavia University...")
   #convert_mat_to_npy(
   #    mat_data_path=os.path.join(BASE, "Pavia University", "PaviaU.mat"),
   #    mat_label_path=os.path.join(BASE, "Pavia University", "PaviaU_gt.mat"),
   #    out_data_path=os.path.join(BASE, "Pavia University", "paviaUarray.npy"),
   #    out_label_path=os.path.join(BASE, "Pavia University", "PUgt.npy"),
   #)
    print("Converting KSC")
    convert_mat_to_npy(
        mat_data_path=os.path.join(BASE, "KSC", "KSC.mat"),
        mat_label_path=os.path.join(BASE, "KSC", "KSC_gt.mat"),
        out_data_path=os.path.join(BASE, "KSC", "KSC.npy"),
        out_label_path=os.path.join(BASE, "KSC", "KSCgt.npy"),
    )
