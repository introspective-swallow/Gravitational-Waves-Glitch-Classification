import torch
import numpy as np
from torchvision import transforms
from PIL import Image
  



def process_image(im):
    if type(im) != torch.Tensor:
        im = torch.from_numpy(im)

    im = (im-torch.min(im))/torch.max(im)
    return im

def process_image_np(im):
    im = (im-np.min(im))/np.max(im)
    return im


# Define the image transformation to resize the image to 224x224
def process_image_augment_train(image, size=(140, 170)):
    image = torch.tensor(image)
    custom_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(5),
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0),antialias=True),
        transforms.Normalize(mean=[0.5] ,std=[0.5])
    ])
    return custom_transform(image)

def process_image_augment_test(image, size=(140, 170)):
    image = torch.tensor(image)
    custom_transform = transforms.Compose([
        transforms.Resize(size,antialias=True),
        transforms.Normalize(mean=[0.5] ,std=[0.5])
    ])
    return custom_transform(image)

def process_parallel_image_augment_train(image):
    image = torch.tensor(image).unsqueeze(1)
    custom_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(5),
        transforms.RandomResizedCrop((128, 128), scale=(0.7, 1.0),antialias=True),
        transforms.Normalize(mean=[0.5] ,std=[0.5])
    ])
    return custom_transform(image).squeeze(1)

def process_parallel_image_augment_test(image):
    image = torch.tensor(image).unsqueeze(1)
    custom_transform = transforms.Compose([
        transforms.Resize((128,128),antialias=True),
        transforms.Normalize(mean=[0.5] ,std=[0.5])
    ])
    return custom_transform(image).squeeze(1)
