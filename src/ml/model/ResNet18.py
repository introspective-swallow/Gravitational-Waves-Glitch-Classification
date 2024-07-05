import torch.nn as nn
import torchvision.models as models
from ml.model.init_weights import init_weights

# Resnet18 from scratch
def ResNet18(num_classes=22):
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.init_weights = init_weights
    return model

# Resnet18 pretrained 
def ResNet18_3Channel(num_classes=22):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    init_weights(model.fc)
    return model

# Resnet18 pretrained adapted to 1 channel
def ResNet18_pretrained(num_classes=22):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # initialize weights of new layers
    init_weights(model.conv1)        
    init_weights(model.fc)
    
    return model

# Resnet18 pretrained 
def ResNet18_pretrained3Channel(num_classes=22):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # initialize weights of new layers
    init_weights(model.fc)
    
    return model