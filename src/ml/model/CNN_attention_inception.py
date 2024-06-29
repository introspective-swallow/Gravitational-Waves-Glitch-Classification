import torch
from torch import nn
from torch.nn import functional as F
from ml.model.init_weights import init_weights
from ml.model.blocks import conv_blocks, res_blocks
from ml.model.CNN_attention import AttentionModule

# Multiple view fusion layer
class MultiViewsAttentionRes(nn.Module):
    def __init__(self, dropout=0.5, latent_vectors=128, hidden_units=64, arch=((5,128,2,(64, (96, 128), (16, 32), 32)),)):
        super().__init__()
        self.latent_vectors = latent_vectors
        self.hidden_units = hidden_units
        self.arch = arch
        self.dropout = dropout
        self.h1 = self.head()
        self.h2 = self.head()
        self.h3 = self.head()
        self.h4 = self.head()
        self.attention = AttentionModule(latent_vectors=latent_vectors, hidden_units=hidden_units)

    def forward(self, x):
        h1 = self.h1(x[:,0,:,:].unsqueeze(1))
        h2 = self.h2(x[:,1,:,:].unsqueeze(1))
        h3 = self.h3(x[:,2,:,:].unsqueeze(1))
        h4 = self.h4(x[:,3,:,:].unsqueeze(1))
        multi = torch.stack((h1, h2, h3, h4), axis=1)
        return self.attention(multi)
    
    def head(self):
        res_blocks = res_blocks(self.arch)
        return nn.Sequential(
            nn.LazyConv2d(out_channels=128, kernel_size=5), 
            nn.MaxPool2d(kernel_size=2),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),

            *res_blocks,

            nn.LazyConv2d(out_channels=128, kernel_size=5), 
            nn.MaxPool2d(kernel_size=2),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),

            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.LazyLinear(self.latent_vectors),
            nn.ReLU(),
        )

    
# Create model structure
class CNN_attention_res(nn.Module):
    def __init__(self, num_classes=22, dropout = 0.5, latent_vectors=128, hidden_units=64, arch=((5,128,2,(64, (96, 128), (16, 32), 32)),)):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.net = nn.Sequential(
            MultiViewsAttentionRes(latent_vectors=latent_vectors, hidden_units=hidden_units, arch=arch),
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
