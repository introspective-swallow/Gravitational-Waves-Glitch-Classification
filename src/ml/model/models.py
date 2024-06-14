import torch
from torch import nn
import warnings

# Initialize data
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.dirac_(m.weight)

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


# ------------------------ Gravity Spy models

# 1xCNN
class GS_CNN(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()
        self.num_classes = num_classes
        with warnings.catch_warnings(): # Hide Lazy modules warning
            warnings.simplefilter("ignore")
            self.net = nn.Sequential(
                nn.LazyConv2d(out_channels=8, kernel_size=5), nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(0.5),
                nn.Flatten(),
                nn.LazyLinear(64), nn.ReLU(),
                nn.Dropout(0.5),
                nn.LazyLinear(num_classes))    
            
    def forward(self, x):
        return self.net(x)

    def init_weights(self, dummy_input):
        self.forward(dummy_input)
        self.net.apply(init_weights)


# 3xCNN
class GS_CNN3(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            nn.LazyConv2d(out_channels=64, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.LazyLinear(256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(num_classes))    
    def forward(self, x):
        return self.net(x)

    def init_weights(self, dummy_input):
        self.forward(dummy_input)
        self.net.apply(init_weights)

# -- CNN parallel

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
        return nn.Sequential(
                nn.LazyConv2d(out_channels=128, kernel_size=5), 
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
    )

class CNN_parallel(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()
        self.num_classes = num_classes
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