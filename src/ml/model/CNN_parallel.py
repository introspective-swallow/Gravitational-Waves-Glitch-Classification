import torch
from torch import nn
from ml.model.init_weights import init_weights
import warnings


# Multiple view fusion layer
class MultiViews(nn.Module):
    def __init__(self, arch, dropout):
        super().__init__()
        self.dropout = dropout
        self.arch = arch
        self.h1 = self.head()
        self.h2 = self.head()
        self.h3 = self.head()
        self.h4 = self.head()

    def forward(self, x):
        h1 = self.h1(x[:,0,:,:].unsqueeze(1))
        h2 = self.h2(x[:,1,:,:].unsqueeze(1))
        h3 = self.h3(x[:,2,:,:].unsqueeze(1))
        h4 = self.h4(x[:,3,:,:].unsqueeze(1))
        return torch.cat((h1, h2, h3, h4), axis=1)
    
    def head(self):
        conv_blks = []
        for (kernel_size, out_channels, pad) in self.arch:
            conv_blks.append(nn.Sequential(
                nn.LazyConv2d(out_channels=out_channels, kernel_size=kernel_size, padding=pad),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                )
            )
        return nn.Sequential(*conv_blks)

class CNN_parallel(nn.Module):
    def __init__(self, num_classes=22, dropout=0.5, arch1=((5, 128, 0),), arch2=((5, 128, 0),)):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        with warnings.catch_warnings(): # Hide Lazy modules warning
            warnings.simplefilter("ignore")
            conv_blks = []
            for (kernel_size, out_channels, pad) in arch2:
                conv_blks.append(nn.Sequential(
                    nn.LazyConv2d(out_channels=out_channels, kernel_size=kernel_size, padding=pad),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    )
                )
            self.net = nn.Sequential(
                MultiViews(arch1, dropout),
                *conv_blks,
                nn.Flatten(),
                nn.Dropout(self.dropout),
                nn.LazyLinear(256), nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.LazyLinear(256), nn.ReLU(),
                nn.LazyLinear(num_classes)    
            )  