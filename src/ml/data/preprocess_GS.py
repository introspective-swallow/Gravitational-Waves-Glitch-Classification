import torch
import numpy as np
from torchvision import transforms
from PIL import Image
  
custom_transform_train = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop((140, 170), scale=(0.7, 1.0),antialias=True),
    transforms.Normalize(mean=[0.5] ,std=[0.5])
])

custom_transform_test = transforms.Compose([
    transforms.Resize((140,170),antialias=True),
    transforms.Normalize(mean=[0.5] ,std=[0.5])
])

custom_transform_train128 = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop((128, 128), scale=(0.7, 1.0),antialias=True),
    transforms.Normalize(mean=[0.5] ,std=[0.5])
])

custom_transform_test128 = transforms.Compose([
    transforms.Resize((128, 128),antialias=True),
    transforms.Normalize(mean=[0.5] ,std=[0.5])
])



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
    return custom_transform_train(image).squeeze(1)

def process_parallel_image_augment_test(image):
    image = torch.tensor(image).unsqueeze(1)
    return custom_transform_test(image).squeeze(1)

# Define the image transformation to resize the image to 224x224
def process_parallel_image_augment_train_post(image):
    image = torch.tensor(image)
    # Apply to each view
    print(image.shape)
    for i in range(4):
        image[i,:,:] = process_image_augment_train(image[i])
    return image

def process_parallel_image_augment_test_post(image):
    image = torch.tensor(image)
    for i in range(4):
        image[i,:,:] = process_image_augment_train(image[i])
    return image


def process_merged_image_augment_train_post(image):
    image = torch.tensor(image)
    # Apply to each view
    image[:,:140,:170] = custom_transform_train(image[:,:140,:170])
    image[:,140:,:170] = custom_transform_train(image[:,140:,:170])
    image[:,:140,170:] = custom_transform_train(image[:,:140,170:])
    image[:,140:,170:] = custom_transform_train(image[:,140:,170:])
    return image

def process_merged_image_augment_test_post(image):
    image = torch.tensor(image)
    # Apply to each view
    image[:,:140,:170] = custom_transform_test(image[:,:140,:170])
    image[:,140:,:170] = custom_transform_test(image[:,140:,:170])
    image[:,:140,170:] = custom_transform_test(image[:,:140,170:])
    image[:,140:,170:] = custom_transform_test(image[:,140:,170:])
    return image

def process_merged_image_augment_train_post128(image):
    image = torch.tensor(image)
    # Apply to each view
    image[:,:128,:128] = custom_transform_train128(image[:,:128,:128])
    image[:,128:,:128] = custom_transform_train128(image[:,128:,:128])
    image[:,:128,128:] = custom_transform_train128(image[:,:128,128:])
    image[:,:128:,128:] = custom_transform_train128(image[:,:128:,128:])
    return image

def process_merged_image_augment_test_post128(image):
    image = torch.tensor(image)
    # Apply to each view
    image[:,:128,:128] = custom_transform_test128(image[:,:128,:128])
    image[:,128:,:128] = custom_transform_test128(image[:,128:,:128])
    image[:,:128,128:] = custom_transform_test128(image[:,:128,128:])
    image[:,:128:,128:] = custom_transform_test128(image[:,:128:,128:])
    return image

def process_merged_image_augment_train_post3channel128(image):
    image = torch.tensor(image)
    proc_im = torch.zeros(image.shape[0], 256,256)
    # Apply to each view
    proc_im[:,:128,:128] = custom_transform_train128(image[:,:140,:170])
    proc_im[:,128:,:128] = custom_transform_train128(image[:,140:,:170])
    proc_im[:,:128,128:] = custom_transform_train128(image[:,:140,170:])
    proc_im[:,128:,128:] = custom_transform_train128(image[:,140:,170:])
    proc_im = torch.cat((proc_im, proc_im, proc_im), axis=0)

    return proc_im

def process_merged_image_augment_test_post3channel128(image):
    image = torch.tensor(image)
    proc_im = torch.zeros(image.shape[0], 256,256)
    # Apply to each view
    proc_im[:,:128,:128] = custom_transform_test128(image[:,:140,:170])
    proc_im[:,128:,:128] = custom_transform_test128(image[:,140:,:170])
    proc_im[:,:128,128:] = custom_transform_test128(image[:,:140,170:])
    proc_im[:,128:,128:] = custom_transform_test128(image[:,140:,170:])
    proc_im = torch.cat((proc_im, proc_im, proc_im), axis=0)
    return proc_im