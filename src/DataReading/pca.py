
import spectral
import numpy as np
import matplotlib.pyplot as plt
from spectral import open_image

# Load the image
hsi_array = np.load(
    r"D:/Univali/TCC3/ANN_Models_Hyperspectral_Images/src/Datasets/Indian_Pines/indianpinearray.npy"
)

hsi_array = hsi_array[:, :, :-1]
print(hsi_array.shape)
print(hsi_array.shape)


# %%
# Visualize a band
band = hsi_array[:, :, 150]  # Example band index
plt.imshow(band, cmap='gray')
plt.title('Band')
plt.show()

# %%
# Plot the spectral signature for a specific pixel
pixel = hsi_array[100, 100, :]  # Extract spectral signature
pixel = np.squeeze(pixel)
plt.plot(pixel, marker='o')  # Add markers for better visibility
plt.title('Spectral Signature at (100, 100)')
plt.xlabel('Band Index')
plt.ylabel('Reflectance')
plt.grid(True)
plt.show()

# %%
from sklearn.preprocessing import MinMaxScaler

# Normalize the image
def normalize_image(image):
    reshaped = image.reshape(-1, image.shape[-1])
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(reshaped).reshape(image.shape)
    return normalized

normalized_hsi = normalize_image(hsi_array)

# %%
from sklearn.decomposition import PCA

# Perform PCA
reshaped = hsi_array.reshape(-1, hsi_array.shape[-1])
pca = PCA(n_components=3)
reduced = pca.fit_transform(reshaped)

# Reshape back
reduced_image = reduced.reshape(
    hsi_array.shape[0],
    hsi_array.shape[1],
    3
)

pca_vis = np.zeros_like(reduced_image, dtype=np.float32)

for i in range(3):
    channel = reduced_image[:, :, i]
    pca_vis[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())
plt.imshow(pca_vis)
plt.title("PCA RGB (Normalized for Visualization)")
plt.axis("off")
plt.show()
plt.show()