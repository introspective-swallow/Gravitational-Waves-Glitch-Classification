import torch.nn as nn
import torchvision.models as models
from ml.model.init_weights import init_weights


# Resnet18 from scratch
def vit_b_16(num_classes=22):
    model = models.vit_b_16()
    model.conv_proj = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    model.init_weights = init_weights
    return model

# Resnet18 pretrained adapted to 1 channel
def vit_b_16_pretrained(num_classes=22):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    model.conv_proj = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # initialize weights of new layers
    init_weights(model.conv_proj)        
    init_weights(model.heads.head)
    
    return model

# Resnet18 pretrained 
def vit_b_16_pretrained3Channel(num_classes=22):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # initialize weights of new layers
    init_weights(model.heads.head)
    
    return model