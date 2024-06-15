import torch
from torch import nn
from ml.model.init_weights import init_weights
import warnings

# Multiple view fusion layer
class MultiViews(nn.Module):
    def __init__(self,):
        super().__init__()
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
        with warnings.catch_warnings(): # Hide Lazy modules warning
            warnings.simplefilter("ignore")
            return nn.Sequential(
                    nn.LazyConv2d(out_channels=128, kernel_size=5), 
                    nn.MaxPool2d(kernel_size=2),
                    nn.ReLU(),
        )

class CNN_parallel(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()
        self.num_classes = num_classes
        with warnings.catch_warnings(): # Hide Lazy modules warning
            warnings.simplefilter("ignore")
            self.net = nn.Sequential(
                MultiViews(),
                nn.LazyConv2d(out_channels=128, kernel_size=5), 
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(256), nn.ReLU(),
                nn.LazyLinear(num_classes)    
            )  
        
    def forward(self, x):
        return self.net(x)

    def init_weights(self, dummy_input):
        self.forward(dummy_input)
        self.net.apply(init_weights)
