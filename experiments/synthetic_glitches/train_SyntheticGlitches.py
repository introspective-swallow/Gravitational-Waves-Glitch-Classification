# Import libreries
import pandas as pd
import numpy as np
import os
import random
import time
from tqdm import tqdm
from pathlib import Path

from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2

from src.data.preprocess import SG_process_image as process_image
from src.model.models import SG_CNN as CNN
from src.data.datasets import SG_WaveFormDatasetFast as WaveFormDataset, get_raw_SG_dataset
from src.training.training import train_loop, test_loop

# Get device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

DATA = Path("./")
OUTPUT = Path("./")

# Hyperparameters
batch_size = 30
lr = 0.001
weight_decay = 2*1e-4
params = {"batch_size":[batch_size], "lr":[lr], "weight_decay":[weight_decay]}
df = pd.DataFrame.from_dict(params)
df.to_csv("/kaggle/working/paper-model-params.csv")

# Get data

train_annotations, train_files, val_annotations, val_files, test_annotations, test_files = get_raw_SG_dataset(DATA)

training_data = WaveFormDataset(train_annotations, train_files, transform=process_image)
val_data = WaveFormDataset(val_annotations, val_files, split="val", transform=process_image)
test_data = WaveFormDataset(test_annotations, test_files, split="test", transform=process_image)

# Define dataloaders
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Instantiate model
model = CNN()
model= nn.DataParallel(model)
model.to(device)

# Initialize model
model.module.init_weights(next(iter(train_dataloader))[0].to(device))

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)

# Train model
epochs = 200
train_log = []
test_log = []
best_acc = 0
test_acc_log = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    
    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, epoch=epoch, max_epochs=epochs)
    test_loss, test_acc = test_loop(val_dataloader, model, loss_fn, split="val")
    train_log.append(train_loss)
    test_log.append(test_loss)
    test_acc_log.append(test_acc)
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            'accuracy': test_acc
            }, OUTPUT/f'CNN.pt')
print("Done!")
df = pd.DataFrame({"train_log":train_log, "test_log":test_log, "test_acc_log":test_acc_log})
df.to_csv(OUTPUT/f"CNN.csv")
print("Best val acc: ", best_acc)

model = CNN()
model= nn.DataParallel(model)
model.to(device)
model.module.init_weights(next(iter(train_dataloader))[0].to(device))

checkpoint = torch.load(OUTPUT/f'CNN.pt')
print("Best model from epoch ", checkpoint["epoch"])
model.load_state_dict(checkpoint['model_state_dict'])

loss, accuracy = test_loop(test_dataloader, model, loss_fn, split="test")
