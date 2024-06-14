import pandas as pd
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm

# ------------ Synthetic glitches datasets

def get_raw_SG_dataset(data_path):
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



# Define dataset class for the waveforms
class SG_WaveFormDataset(Dataset):
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
class SG_WaveFormDatasetFast(Dataset):
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

 
# ------------ Gravity Spy datasets
def get_metadata(data_file_path, meta_file_path):
    with h5py.File(data_file_path, 'r') as f:
        names_classes = list(f.keys())
        num_classes = len(names_classes)
        splits = list(f[names_classes[0]].keys())
        file_name_test = list(f[names_classes[0]][splits[0]].keys())[0]
        test = f[names_classes[0]][splits[0]][file_name_test]
        resolutions = list(test)
    dataset = pd.read_csv(meta_file_path)
    return dataset, names_classes, resolutions   

# Define dataset class for the waveforms
class GS_WaveFormDataset(Dataset):
    def __init__(self, images, labels, split='train', transform=None, device="cpu"):
        self.split = split
        self.images = torch.tensor(images).to(device)
        self.labels = torch.tensor(labels).to(device)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

    
class GS_WaveFormDatasetFast(Dataset):
    def __init__(self, data_file_path, meta_file_path, split, res=0.5, transform=None, device="cpu"):
        self.split = split
        self.labels = []
        self.transform = transform
        self.images = []
        
        complete_dataset, classes, resolutions = get_metadata(data_file_path, meta_file_path)
        dataset = complete_dataset[complete_dataset["sample_type"]==split].reset_index(drop=True)

        with h5py.File(data_file_path, 'r') as f:
            pbar = tqdm(range(len(dataset)))
            pbar.set_description(f"Loading {split} data")
            for i in pbar:
                id = dataset["gravityspy_id"][i]

                label = dataset["label"][i]
                if(res==0.5):
                    image = self.transform(np.array(f[label][split][id][resolutions[0]]))
                elif(res==1):
                    image = self.transform(np.array(f[label][split][id][resolutions[1]]))
                elif(res==2):
                    image = self.transform(np.array(f[label][split][id][resolutions[2]]))
                elif(res==4):
                    image = self.transform(np.array(f[label][split][id][resolutions[3]]))
                self.images.append(image)
                self.labels.append(classes.index(label))
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label

class GS_WaveFormDatasetParallel(Dataset):
    def __init__(self, data_file_path, meta_file_path, split, transform=None, device="cpu"):
        self.split = split
        self.labels = []
        self.transform = transform
        self.images = []

        dataset, classes, resolutions = get_metadata(data_file_path, meta_file_path)

        with h5py.File(data_file_path, 'r') as f:
            for i in tqdm(range(len(dataset))):
                id = dataset["gravityspy_id"][i]
                split_sample = dataset["sample_type"][i]
                if split_sample == split:
                    label = dataset["label"][i]
                    image05 = self.transform(np.array(f[label][split][id][resolutions[0]]))
                    image10 = self.transform(np.array(f[label][split][id][resolutions[1]]))
                    image20 = self.transform(np.array(f[label][split][id][resolutions[2]]))
                    image40 = self.transform(np.array(f[label][split][id][resolutions[3]]))
                    image_multi = torch.cat((image05, image10, image20, image40), axis=0)
                    self.images.append(image_multi)
                    self.labels.append(classes.index(label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label    

class GS_WaveFormDatasetMerged(Dataset):
    def __init__(self, data_file_path, meta_file_path, split, transform=None, device="cpu"):
        self.split = split
        self.labels = []
        self.transform = transform
        self.images = []

        dataset, classes, resolutions = get_metadata(data_file_path, meta_file_path)

        with h5py.File(data_file_path, 'r') as f:
            for i in tqdm(range(len(dataset))):
                id = dataset["gravityspy_id"][i]
                split_sample = dataset["sample_type"][i]
                if split_sample == split:
                    label = dataset["label"][i]
                    image05 = self.transform(np.array(f[label][split][id][resolutions[0]]))
                    image10 = self.transform(np.array(f[label][split][id][resolutions[1]]))
                    image20 = self.transform(np.array(f[label][split][id][resolutions[2]]))
                    image40 = self.transform(np.array(f[label][split][id][resolutions[3]]))
                    image_multi1 = torch.cat((image05, image10), axis=1)
                    image_multi2 = torch.cat((image20, image40), axis=1)
                    image_multi = torch.cat((image_multi1, image_multi2), axis=2)

                    self.images.append(image_multi)
                    self.labels.append(classes.index(label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label

 