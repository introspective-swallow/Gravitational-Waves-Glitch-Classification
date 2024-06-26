import pandas as pd
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm

def get_raw_dataset(data_path):
    # Define split and class names
    split_names = ['train','val','test']
    noise_names = ['GAUSS','CHIRPLIKE','RD','SCATTEREDLIKE','SG','WHISTLELIKE','NOISE']

    # Create image filepaths lists and annotations lists
    train_files = []
    train_annotations = []
    val_files = []
    val_annotations = []
    test_files = []
    test_annotations = []

    for split in split_names:
        for noise in noise_names:
            dir_path = os.path.join(data_path, 'figshare_image_dataset_np_'+noise+'_'+split, split,noise)
            files = os.listdir(dir_path)
            files = [os.path.join(dir_path, file) for file in files]
            for file in files:
                if not os.path.exists(file):
                    print("File not found!")
                    exit(1)
            if split == 'train':
                train_files.extend(files)
                train_annotations.extend([noise] * len(files))

            elif split == 'val':
                val_files.extend(files)
                val_annotations.extend([noise] * len(files))

            elif split == 'test':
                test_files.extend(files)
                test_annotations.extend([noise] * len(files))

    # Transform annotations to integers
    train_annotations = [noise_names.index(noise) for noise in train_annotations]
    val_annotations = [noise_names.index(noise) for noise in val_annotations]
    test_annotations = [noise_names.index(noise) for noise in test_annotations]
    return train_annotations, train_files, val_annotations, val_files, test_annotations, test_files

def get_dataset(annotations_df, img_filepath_list, transform=None):
    images = []
    image_labels = annotations_df
    for file_path in tqdm(img_filepath_list):
        image = np.load(file_path)
        if transform:
            transformed_image = transform(image)
        images.append(transformed_image)
    return images, image_labels


# Define dataset class for the waveforms
class WaveFormDataset(Dataset):
    def __init__(self, annotations_df, img_filepath_list, split='train', transform=None, target_transform=None):
        self.split = split
        self.img_labels = annotations_df
        self.img_filepath_list = img_filepath_list
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_filepath_list[idx]
        image = np.load(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)
        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = label
        return image, label

# Define dataset class for the waveforms
class WaveFormDatasetFast(Dataset):
    def __init__(self, annotations_df, img_filepath_list, split='train', transform=None):
        self.split = split
        self.img_labels = annotations_df
        self.transform = transform
        self.img_filepath_list = img_filepath_list
        self.images = []
        for file_path in tqdm(self.img_filepath_list):
            image = np.load(file_path)
            transformed_image = self.transform(image)
            self.images.append(transformed_image)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.img_labels[idx]
        return image, label