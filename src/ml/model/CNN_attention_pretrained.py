import torch
from torch import nn
from torch.nn import functional as F
from ml.model.init_weights import init_weights
from ml.model.blocks import conv_blocks
from ml.utils.utils import load_model
from ml.model.CNN import CNN


# Resnet18 pretrained adapted to 1 channel
def CNN_pretrained(num_classes=22, PATH="", dummy_dims=(64, 1, 140, 170), device="cuda"):
    model = load_model(CNN, PATH / "model.pt", dummy_dims=dummy_dims, params={}, device=device)

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    model.net[-1] = nn.Linear(model.net[-1].in_features, num_classes)
    
    # initialize weights of new layers
    init_weights(model.net[-1])
    
    return model



class MultiViewsAttention(nn.Module):
    def __init__(self, latent_vectors=128, hidden_units=64, dropout=0.25, arch=((5,128)), device="cuda",
                  dummy_dims=(64, 1, 140, 170), PATH_05="", PATH_1="", PATH_2="", PATH_4=""):
        super().__init__()
        self.latent_vectors = latent_vectors
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.arch = arch
        self.device = device
        self.h1, _ =  CNN_pretrained(PATH=PATH_05, dummy_dims=dummy_dims, device=self.device)
        self.h2, _ =  CNN_pretrained(PATH=PATH_1, dummy_dims=dummy_dims, device=self.device)
        self.h3, _ =  CNN_pretrained(PATH=PATH_2, dummy_dims=dummy_dims, device=self.device)
        self.h4, _ =  CNN_pretrained(PATH=PATH_4, dummy_dims=dummy_dims, device=self.device)

        self.attention = AttentionModule(latent_vectors=latent_vectors, hidden_units=hidden_units)

    def forward(self, x):
        h1 = self.h1(x[:,0,:,:].unsqueeze(1))
        h2 = self.h2(x[:,1,:,:].unsqueeze(1))
        h3 = self.h3(x[:,2,:,:].unsqueeze(1))
        h4 = self.h4(x[:,3,:,:].unsqueeze(1))
        multi = torch.stack((h1, h2, h3, h4), axis=1)
        return self.attention(multi)
    
    def head(self, arch=((5,128))):
        conv_blks =conv_blocks(arch)
        
        return nn.Sequential(
                *conv_blks,
                nn.Flatten(),
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
    def __init__(self, dropout=0.25, num_classes=22, arch=((5,128),(5,128)), latent_vectors=128, hidden_units=64,
                 dummy_dims=(64, 1, 140, 170), PATH_05="", PATH_1="", PATH_2="", PATH_4=""):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.arch = arch
        self.net = nn.Sequential(
            MultiViewsAttention(dropout=dropout, arch=arch, latent_vectors=latent_vectors, hidden_units=hidden_units,
                                dummy_dims=dummy_dims, PATH_05=PATH_05, PATH_1=PATH_1, PATH_2=PATH_2, PATH_4=PATH_4),
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