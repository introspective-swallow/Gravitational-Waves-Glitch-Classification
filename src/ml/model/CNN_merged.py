import torch
from torch import nn
from ml.model.init_weights import init_weights

# -- CNN merged
class CNNmerged(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=128, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.LazyConv2d(out_channels=128, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.LazyLinear(256), nn.ReLU(),
            nn.Dropout(0.25),
            nn.LazyLinear(num_classes))    
    def forward(self, x):
        return self.net(x)

    def init_weights(self, dummy_input):
        self.forward(dummy_input)
        self.net.apply(init_weights)