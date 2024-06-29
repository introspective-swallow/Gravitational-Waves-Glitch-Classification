import torch.nn as nn

def conv_block(kernel_size, out_channels, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

def conv_blocks(arch):
    layers = []
    for params in arch:
        if len(params) == 2:
            kernel_size, out_channels = params
            padding = 0
        else:
            kernel_size, out_channels, padding = params
        layers.append(conv_block(kernel_size, out_channels, padding))
    return nn.Sequential(*layers)

class Inception(nn.Module):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super().__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)


def res_block(kernel_size=5, out_channels=128, padding=0, inception = (64, (96, 128), (16, 32), 32)):
    return nn.Sequential(
        nn.LazyConv2d(out_channels=out_channels, kernel_size=kernel_size, padding=padding), 
        nn.MaxPool2d(kernel_size=2),
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
        Inception(*inception)
    )

def res_blocks(arch):
    layers = []
    for params in arch:
        if len(params) == 3:
            kernel_size, out_channels, inception = params
            padding = 0
        else:
            kernel_size, out_channels, padding, inception = params
        layers.append(res_block(kernel_size, out_channels, padding, inception = inception))
    return nn.Sequential(*layers)

def vgg_block(kernel_size=3, out_channels=64, padding=2):
    return nn.Sequential(
        nn.LazyConv2d(out_channels=out_channels, kernel_size=kernel_size, padding=padding), nn.ReLU(),
        nn.LazyConv2d(out_channels=out_channels, kernel_size=kernel_size, padding=padding), nn.ReLU(),
        nn.LazyConv2d(out_channels=out_channels, kernel_size=kernel_size, padding=padding), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

def vgg_blocks(arch):
    layers = []
    for params in arch:
        if len(params) == 2:
            kernel_size, out_channels = params
            padding = 0
        else:
            kernel_size, out_channels, padding = params
        layers.append(vgg_block(kernel_size, out_channels, padding))
    return nn.Sequential(*layers)
