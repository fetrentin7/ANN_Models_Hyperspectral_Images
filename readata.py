import numpy as np
import pandas as pd
import os

data = np.load("Datasets/Indian_Pines/indianpinearray.npy")

#print(f"Data type: {data.dtype}")
print(f"Data shape: {data.shape}")

first_element = data.flat[0]
print(f"The first element is: {first_element}")