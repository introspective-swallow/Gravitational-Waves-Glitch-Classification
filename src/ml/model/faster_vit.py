import torch.nn as nn
from fastervit import create_model
from ml.model.init_weights import init_weights



# Resnet18 pretrained adapted to 1 channel
def faster_vit_6_224(num_classes=22):
    model = create_model('faster_vit_6_224', 
                          pretrained=True)
    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    model.head = nn.Linear(model.head.in_features, num_classes)
    
    # initialize weights of new layers
    init_weights(model.head)
    
    return model
