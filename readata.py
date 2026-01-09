import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

data = np.load("Datasets/Indian_Pines/indianpinearray.npy", mmap_mode='r')

#print(f"Data type: {data.dtype}") #checks the data type
print(f"Data shape: {data.shape}") #returns a tuple (number of rows/columns)

"""first_element = data.flat[0] #1D iterator over thhe entire array row
    print(f"The first element is: {first_element}")

for i in data:
    for element in i:
        print(element) #matrix normal iteration
for i in data[:10]: #accessing the first 10 elements
    print(i) """

#print(f"data shape: {data.shape}")
#plt.imshow(data, cmap='gray')
#plt.title('Visualization of a Single Band')
#plt.show()

band = data[:, :, 0]
plt.imshow(band, cmap='viridis')
plt.title('Visualization of Band 1')
plt.show()

rgb_data = data[:, :, [30, 20, 10]]  # R, G, B

# Normalização simples (0–1)
if data.shape[2] >= 3:
    rgb_data = data[:, :, :3]

    rgb_min = rgb_data.min()
    rgb_max = rgb_data.max()
    rgb_data = (rgb_data - rgb_min) / (rgb_max - rgb_min)

    plt.imshow(rgb_data)
    plt.title("Pseudo-RGB image (first 3 bands)")
    plt.axis('off')
    plt.show()