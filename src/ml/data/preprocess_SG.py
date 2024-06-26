import torch
import numpy as np
from torchvision.transforms import v2
  
def normalize_percentile(im):
    # Normalize pixel values between 5-95th percentiles
    idx = torch.where((torch.quantile(im,0.05) < im) * (torch.quantile(im,0.95) > im))
    im[idx] = im[idx] / torch.median(im[idx])
    return im


def process_image(im):
    # Transform to 32-bit gray scale
    im = torch.from_numpy(im)
    im = (im-torch.min(im))/torch.max(im)
    im = 32*torch.log2(im+1e-9)
        
    # Normalize values between 5-95 percentiles
    im = normalize_percentile(im)
        
    # Normalize each frequency by its median value
    #im = im / np.median(im, axis=1, keepdims=True)
    
    # Scale by factor of 0.55
    im = v2.Resize(size=(282,132),antialias=True)(im.unsqueeze(0)).squeeze(0).float()
    im = im.unsqueeze(0)
    return im
