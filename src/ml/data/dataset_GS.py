import pandas as pd
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm

def get_metadata(data_file_path, meta_file_path):
    with h5py.File(data_file_path, 'r') as f:
        names_classes = list(f.keys())
        num_classes = len(names_classes)
        splits = list(f[names_classes[0]].keys())
        file_name_test = list(f[names_classes[0]][splits[0]].keys())[0]
        test = f[names_classes[0]][splits[0]][file_name_test]
        resolutions = list(test)
    dataset = pd.read_csv(meta_file_path)
    class_counts = [sum(dataset["label"]==glitch) for glitch in names_classes] 
    return {"names_classes": names_classes, "num_classes": num_classes, "splits": splits, "resolutions": resolutions, "dataset": dataset, "class_counts": class_counts}

# Define dataset class for the waveforms
class WaveFormDatasetPost(Dataset):
    def __init__(self, data_file_path, meta_file_path, split, res, transform=None, device="cpu", save_to_ram=False):
        self.split = split
        self.transform = transform
        self.labels = []
        self.images = []

        metadata = get_metadata(data_file_path, meta_file_path)
        classes = metadata["names_classes"]
        resolutions = metadata["resolutions"]
        complete_dataset = metadata["dataset"]

        dataset = complete_dataset[complete_dataset["sample_type"]==split].reset_index(drop=True)

        with h5py.File(data_file_path, 'r') as f:
            pbar = tqdm(range(len(dataset)))
            pbar.set_description(f"Loading {split} data")
            for i in pbar:
                id = dataset["gravityspy_id"][i]

                label = dataset["label"][i]
                if(res==0.5):
                    image = np.array(f[label][split][id][resolutions[0]])
                elif(res==1):
                    image = np.array(f[label][split][id][resolutions[1]])
                elif(res==2):
                    image = np.array(f[label][split][id][resolutions[2]])
                elif(res==4):
                    image = np.array(f[label][split][id][resolutions[3]])
                self.images.append(image)
                self.labels.append(classes.index(label))


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

    
class WaveFormDataset(Dataset):
    def __init__(self, data_file_path, meta_file_path, split, res=0.5, transform=None, device="cpu", save_to_ram=True):
        self.split = split
        self.transform = transform
        self.labels = []
        self.images = []
        self.save_to_ram = save_to_ram

        metadata = get_metadata(data_file_path, meta_file_path)
        complete_dataset = metadata["dataset"]
        classes = metadata["names_classes"]
        resolutions = metadata["resolutions"]
        
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
        if save_to_ram:
            self.labels = torch.tensor(self.labels).to(device)
            self.images = torch.stack(self.images).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.save_to_ram:
            image = self.images[idx,:,:,:]
        else:
            image = self.images[idx]
        label = self.labels[idx]

        return image, label

class WaveFormDatasetParallelPost(Dataset):
    def __init__(self, data_file_path, meta_file_path, split, transform=None, device="cpu", save_to_ram=False):
        self.split = split
        self.transform = transform
        self.labels = []
        self.images = []

        metadata = get_metadata(data_file_path, meta_file_path)
        complete_dataset = metadata["dataset"]
        classes = metadata["names_classes"]
        resolutions = metadata["resolutions"]
        
        dataset = complete_dataset[complete_dataset["sample_type"]==split].reset_index(drop=True)

        with h5py.File(data_file_path, 'r') as f:
            pbar = tqdm(range(len(dataset)))
            pbar.set_description(f"Loading {split} data")
            for i in pbar:
                id = dataset["gravityspy_id"][i]
                label = dataset["label"][i]
                image05 = np.array(f[label][split][id][resolutions[0]])
                image10 = np.array(f[label][split][id][resolutions[1]])
                image20 = np.array(f[label][split][id][resolutions[2]])
                image40 = np.array(f[label][split][id][resolutions[3]])
                image_multi = np.concatenate((image05, image10, image20, image40), axis=0)
                self.images.append(image_multi)
                self.labels.append(classes.index(label))

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

class WaveFormDatasetParallel(Dataset):
    def __init__(self, data_file_path, meta_file_path, split, transform=None, device="cpu", save_to_ram=True):
        self.split = split
        self.transform = transform
        self.labels = []
        self.images = []
        self.save_to_ram = save_to_ram
        self.device = device
        
        metadata = get_metadata(data_file_path, meta_file_path)
        complete_dataset = metadata["dataset"]
        classes = metadata["names_classes"]
        resolutions = metadata["resolutions"]
        
        dataset = complete_dataset[complete_dataset["sample_type"]==split].reset_index(drop=True)

        with h5py.File(data_file_path, 'r') as f:
            pbar = tqdm(range(len(dataset)))
            pbar.set_description(f"Loading {split} data")
            for i in pbar:
                id = dataset["gravityspy_id"][i]
                label = dataset["label"][i]
                image05 = self.transform(np.array(f[label][split][id][resolutions[0]]))
                image10 = self.transform(np.array(f[label][split][id][resolutions[1]]))
                image20 = self.transform(np.array(f[label][split][id][resolutions[2]]))
                image40 = self.transform(np.array(f[label][split][id][resolutions[3]]))
                image_multi = torch.cat((image05, image10, image20, image40), axis=0)
                self.images.append(image_multi)
                self.labels.append(classes.index(label))

        if self.save_to_ram:
            self.labels = torch.tensor(self.labels).to(device)
            self.images = torch.stack(self.images).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.save_to_ram:
            image = self.images[idx,:,:,:]
        else:
            image = self.images[idx]
        
        label = self.labels[idx]
        return image, label

class WaveFormDatasetMergedPost(Dataset):
    def __init__(self, data_file_path, meta_file_path, split, transform=None, device="cpu", save_to_ram=False):
        self.split = split
        self.transform = transform
        self.labels = []
        self.images = []

        metadata = get_metadata(data_file_path, meta_file_path)
        complete_dataset = metadata["dataset"]
        classes = metadata["names_classes"]
        resolutions = metadata["resolutions"]
        
        dataset = complete_dataset[complete_dataset["sample_type"]==split].reset_index(drop=True)

        with h5py.File(data_file_path, 'r') as f:
            pbar = tqdm(range(len(dataset)))
            pbar.set_description(f"Loading {split} data")
            for i in pbar:
                id = dataset["gravityspy_id"][i]
                label = dataset["label"][i]
                image05 = np.array(f[label][split][id][resolutions[0]])
                image10 = np.array(f[label][split][id][resolutions[1]])
                image20 = np.array(f[label][split][id][resolutions[2]])
                image40 = np.array(f[label][split][id][resolutions[3]])
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

        if self.transform:
            image = self.transform(image)
        
        return image, label

 
class WaveFormDatasetMerged(Dataset):
    def __init__(self, data_file_path, meta_file_path, split, transform=None, device="cpu", save_to_ram=True):
        self.split = split
        self.transform = transform
        self.labels = []
        self.images = []
        self.save_to_ram = save_to_ram

        metadata = get_metadata(data_file_path, meta_file_path)
        complete_dataset = metadata["dataset"]
        classes = metadata["names_classes"]
        resolutions = metadata["resolutions"]

        dataset = complete_dataset[complete_dataset["sample_type"]==split].reset_index(drop=True)

        with h5py.File(data_file_path, 'r') as f:
            pbar = tqdm(range(len(dataset)))
            pbar.set_description(f"Loading {split} data")
            for i in pbar:
                id = dataset["gravityspy_id"][i]
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
        if self.save_to_ram:
            self.labels = torch.tensor(self.labels).to(device)
            self.images = torch.stack(self.images).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.save_to_ram:
            image = self.images[idx,:,:,:]
        else:
            image = self.images[idx]
        
        label = self.labels[idx]
        return image, label
        
class WaveFormDatasetMerged3Channel(Dataset):
    def __init__(self, data_file_path, meta_file_path, split, transform=None, save_to_ram=True, device="cpu"):
        self.split = split
        self.transform = transform
        self.save_to_ram = save_to_ram
        self.labels = []
        self.images = []

        metadata = get_metadata(data_file_path, meta_file_path)
        complete_dataset = metadata["dataset"]
        classes = metadata["names_classes"]
        resolutions = metadata["resolutions"]

        dataset = complete_dataset[complete_dataset["sample_type"]==split].reset_index(drop=True)

        with h5py.File(data_file_path, 'r') as f:
            pbar = tqdm(range(len(dataset)))
            pbar.set_description(f"Loading {split} data")
            for i in pbar:
                id = dataset["gravityspy_id"][i]
                label = dataset["label"][i]
                image05 = self.transform(np.array(f[label][split][id][resolutions[0]]))
                image10 = self.transform(np.array(f[label][split][id][resolutions[1]]))
                image20 = self.transform(np.array(f[label][split][id][resolutions[2]]))
                image40 = self.transform(np.array(f[label][split][id][resolutions[3]]))
                image_multi1 = torch.cat((image05, image10), axis=1)
                image_multi2 = torch.cat((image20, image40), axis=1)
                image_multi = torch.cat((image_multi1, image_multi2), axis=2)

                # Repeat the first channel to have 3 channels
                image_multi = torch.cat((image_multi, image_multi, image_multi), axis=0)

                self.images.append(image_multi)
                self.labels.append(classes.index(label))
        if self.save_to_ram:
            self.labels = torch.tensor(self.labels).to(device)
            self.images = torch.stack(self.images).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.save_to_ram:
            image = self.images[idx,:,:,:]
        else:
            image = self.images[idx]
        
        label = self.labels[idx]
        return image, label

class WaveFormDatasetMerged3ChannelPostprocess(Dataset):
    def __init__(self, data_file_path, meta_file_path, split, transform=None, save_to_ram=True, device="cpu"):
        self.split = split
        self.transform = transform
        self.save_to_ram = save_to_ram
        self.labels = []
        self.images = []

        metadata = get_metadata(data_file_path, meta_file_path)
        complete_dataset = metadata["dataset"]
        classes = metadata["names_classes"]
        resolutions = metadata["resolutions"]

        dataset = complete_dataset[complete_dataset["sample_type"]==split].reset_index(drop=True)

        with h5py.File(data_file_path, 'r') as f:
            pbar = tqdm(range(len(dataset)))
            pbar.set_description(f"Loading {split} data")
            for i in pbar:
                id = dataset["gravityspy_id"][i]
                label = dataset["label"][i]
                image05 = np.array(f[label][split][id][resolutions[0]])
                image10 = np.array(f[label][split][id][resolutions[1]])
                image20 = np.array(f[label][split][id][resolutions[2]])
                image40 = np.array(f[label][split][id][resolutions[3]])
                image_multi1 = torch.cat((image05, image10), axis=1)
                image_multi2 = torch.cat((image20, image40), axis=1)
                image_multi = torch.cat((image_multi1, image_multi2), axis=2)

                # Repeat the first channel to have 3 channels
                image_multi = torch.cat((image_multi, image_multi, image_multi), axis=0)
                image_multi = self.transform(image_multi)

                self.images.append(image_multi)
                self.labels.append(classes.index(label))
        if self.save_to_ram:
            self.labels = torch.tensor(self.labels).to(device)
            self.images = torch.stack(self.images).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.save_to_ram:
            image = self.images[idx,:,:,:]
        else:
            image = self.images[idx]
        
        label = self.labels[idx]
        return image, label