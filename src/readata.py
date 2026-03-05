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

band = data[:, :, 2] #first band

rgb_data = data[:, :, 50]  #RGB

if data.shape[2] >= 3: #at least 3 bands
    rgb_data = data[:, :, :4] #take every row and column of the firsts three elements

    rgb_min = rgb_data.min() #min value of each band
    rgb_max = rgb_data.max()
    rgb_data = (rgb_data - rgb_min) / (rgb_max - rgb_min)


#HYPERSPECTRAL IMAGE
# Subplot 1 - unique band
plt.subplot(1, 2, 1)   # 1 linha, 2 colunas, posição 1
plt.imshow(band, cmap='viridis')
plt.title('Band 0')
plt.axis('off')

#First 4 band using normalizarion
plt.subplot(1, 2, 2)   # 1 linha, 2 colunas, posição 2
plt.imshow(rgb_data)
plt.title('First 4 bands (normalized)')
plt.axis('off')

plt.tight_layout()
plt.show()