import torch.nn as nn

class CNN2D(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.4)

        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Começa com 64, vai para 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.4)
        )

        #self.block4 = nn.Sequential(
        #    nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #    nn.ReLU(inplace=True),
        #    nn.Conv2d(512,512, kernel_size=3, padding=1),
        #    nn.ReLU(inplace=True),
        #)

        #global average ppooling (N, 128, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, classes)
        )
    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x)
        x = self.fc(x)

        return  x
