
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
DATASETS = {
    "IP": {
        "data":   r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/indianpinearray.npy",
        "labels": r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/IPgt.npy",
        "names":  IP_CLASSES
    },
    "PU": {
        "data":   r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Pavia University/paviaUarray.npy",
        "labels": r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Pavia University/PUgt.npy",
        "names":  PU_CLASSES
    },
    "KSC": {
        "data":r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/KSC/KSC.npy",
        "labels": r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/KSC/KSCgt.npy",
        "names":  KSC_CLASSES
    }

    # "SAL": {
    #     "data":   r"D:/.../Salinas/salinasarray.npy",
    #     "labels": r"D:/.../Salinas/Sgt.npy",
    #     "names":  SAL_CLASSES,
    # },

}

def choose_dataset():
    print("Available datasets:", list(DATASETS.keys()))
    dataset = input("Which dataset do you want to train? (e.g. IP, PU, KSC): ").strip().upper()

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(DATASETS.keys())}")

    cfg = DATASETS[dataset]
    return dataset, cfg["data"], cfg["labels"], cfg["names"]