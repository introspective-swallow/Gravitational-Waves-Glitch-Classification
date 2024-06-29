import torch
from torch import nn
from torch.nn import functional as F
from ml.model.init_weights import init_weights
from ml.model.blocks import vgg_blocks

class MultiViewsAttention(nn.Module):
    def __init__(self, latent_vectors=128, hidden_units=64, dropout=0.25, arch=((5,128))):
        super().__init__()
        self.latent_vectors = latent_vectors
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.arch = arch
        self.h1 = self.head(arch=arch)
        self.h2 = self.head(arch=arch)
        self.h3 = self.head(arch=arch)
        self.h4 = self.head(arch=arch)
        self.attention = AttentionModule(latent_vectors=latent_vectors, hidden_units=hidden_units)

    def forward(self, x):
        h1 = self.h1(x[:,0,:,:].unsqueeze(1))
        h2 = self.h2(x[:,1,:,:].unsqueeze(1))
        h3 = self.h3(x[:,2,:,:].unsqueeze(1))
        h4 = self.h4(x[:,3,:,:].unsqueeze(1))
        multi = torch.stack((h1, h2, h3, h4), axis=1)
        return self.attention(multi)
    
    def head(self, arch=((5,128))):
        blocks =vgg_blocks(arch)
        
        return nn.Sequential(
                *blocks,
                nn.Flatten(),
                nn.Dropout(self.dropout),
                nn.LazyLinear(self.latent_vectors),
                nn.ReLU(),
    )

class AttentionModule(nn.Module):
    def __init__(self, latent_vectors = 128, hidden_units = 64):
        super().__init__()
        self.U = nn.Parameter(
            torch.zeros(hidden_units, latent_vectors))
        self.V = nn.Parameter(
            torch.zeros(hidden_units, latent_vectors))
        self.w = nn.Parameter(
            torch.zeros(hidden_units, 1))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.w)

    def forward(self, x):
        result = self.attentionWeight(x[:,0,:]).view(-1, 1)  * x[:,0,:]
        for i in range(1,4):
            result += self.attentionWeight(x[:,i,:]).view(-1, 1)  * x[:,i,:]
        return result
    
    def attentionWeight(self, view):
        x1 = torch.tanh(torch.matmul(self.V, view.T))
        x2 = torch.sigmoid(torch.matmul(self.U, view.T))
        weight = F.softmax(torch.matmul(self.w.T, x1 * x2), dim=1)
        return weight
    
# Create model structure
class CNN_attention(nn.Module):
    def __init__(self, dropout=0.25, num_classes=22, arch=((5,128),(5,128)), latent_vectors=128, hidden_units=64):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.arch = arch
        self.net = nn.Sequential(
            MultiViewsAttention(dropout=dropout, arch=arch, latent_vectors=latent_vectors, hidden_units=hidden_units),
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