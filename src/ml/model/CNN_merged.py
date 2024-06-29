import torch
from torch import nn
from ml.model.init_weights import init_weights
from ml.model.blocks import conv_blocks

import warnings

# -- CNN merged
class CNNmerged(nn.Module):
    def __init__(self, dropout = 0.25, arch=((5,128),(5,128)), num_classes=22):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.arch = arch
        conv_blks = []
        with warnings.catch_warnings(): # Hide Lazy modules warning
            warnings.simplefilter("ignore")
            conv_blks.append(conv_blocks(arch))
            self.net = nn.Sequential(
                *conv_blks,
                nn.Flatten(),
                nn.Dropout(self.dropout),
                nn.LazyLinear(256), nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.LazyLinear(256), nn.ReLU(),
                nn.LazyLinear(num_classes)  
            )
    def forward(self, x):
        return self.net(x)

    def init_weights(self, dummy_input):
        self.forward(dummy_input)
        self.net.apply(init_weights)

