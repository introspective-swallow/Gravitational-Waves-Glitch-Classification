import torch.nn as nn
import torchvision.models as models
from ml.model.init_weights import init_weights

def convnext_tiny(num_classes=22):
    model = models.convnext_tiny()
    model.features[0][0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model.init_weights = init_weights
    return model