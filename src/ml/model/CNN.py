import torch
from torch import nn
import warnings
from ml.model.init_weights import init_weights

# -- 1xCNN
class CNN1(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=16, kernel_size=3), nn.ReLU(),
            nn.LazyConv2d(out_channels=32, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.LazyLinear(512), nn.ReLU(),
            nn.Dropout(0.25),
            nn.LazyLinear(num_classes))    
    def forward(self, x):
        return self.net(x)

    def init_weights(self, dummy_input):
        self.forward(dummy_input)
        self.net.apply(init_weights)
        
# -- 3xCNN
class CNN3(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=16, kernel_size=3), nn.ReLU(),
            nn.LazyConv2d(out_channels=32, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.LazyConv2d(out_channels=64, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.LazyConv2d(out_channels=64, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.LazyConv2d(out_channels=128, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.LazyConv2d(out_channels=128, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.LazyLinear(512), nn.ReLU(),
            nn.Dropout(0.25),
            nn.LazyLinear(num_classes))    
    def forward(self, x):
        return self.net(x)

    def init_weights(self, dummy_input):
        self.forward(dummy_input)
        self.net.apply(init_weights)

# CNN parametrized by architecture

class CNN(nn.Module):
    """
    1xCNN
        arch = ((5,8))
    2xCNN
        arch = ((5,8), (5,16))
    """
    def __init__(self, num_classes=22, dropout=0.5, arch=((5,32), (5,64))):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        conv_blks = []
        for (kernel_size, out_channels) in arch:
            conv_blks.append(nn.Sequential(
                nn.LazyConv2d(out_channels=out_channels, kernel_size=kernel_size), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(dropout)))
            
        self.net = nn.Sequential(
            *conv_blks,
            nn.Flatten(),
            nn.LazyLinear(256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.LazyLinear(num_classes))
        
    def forward(self, x):
        return self.net(x)

    def init_weights(self, dummy_input):
        self.forward(dummy_input)
        self.net.apply(init_weights)


