import spectral
import os
import numpy as np
import scipy.io as sci
import torch
import torch.nn as nn
import torch.optim as optim
from sympy import true
from torch.utils.data import TensorDataset, DataLoader

class CNN2D(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=true),
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.ReLU(inplace=true),
        )

        #global average ppooling (N, 512, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=true),
            nn.Dropout(0.5),
            nn.Linear(512, classes)
        )
    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = self.fc(x)

        return  x

