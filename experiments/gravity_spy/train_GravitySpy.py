from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import h5py
import time
import random
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

from ml.data.preprocess import GS_process_image as process_image
from ml.model.models import GS_CNN as CNN
from ml.data.datasets import GS_WaveFormDatasetFast as WaveFormDataset, get_metadata
from ml.training.training import train_loop, test_loop
from ml.utils.utils import load_model, get_conf_matrix, print_logs, plot_conf_matrix

DATA = Path("./data/gravity_spy")
DATASET_PATH = DATA / "trainingsetv1d0.h5"
METADATA_PATH = DATA / "trainingset_v1d0_metadata.csv"
OUTPUT = Path("./experiments/gravity_spy/exp1")
if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

# Get device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device.")

# Hyperparameters
batch_size = 64
lr = 0.001
weight_decay = 2*1e-4
params = {"batch_size":[batch_size], "lr":[lr], "weight_decay":[weight_decay]}
df = pd.DataFrame.from_dict(params)
df.to_csv(OUTPUT / "params.csv")

# Get data
_, names_classes, _ = get_metadata(DATASET_PATH, METADATA_PATH)

training_data = WaveFormDataset(DATASET_PATH, METADATA_PATH, "train", res=0.5, transform=process_image, device=device)
val_data = WaveFormDataset(DATASET_PATH, METADATA_PATH, "validation", res=0.5, transform=process_image, device=device)
test_data = WaveFormDataset(DATASET_PATH, METADATA_PATH, "test", res=0.5, transform=process_image, device=device)

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
epochs = 1

train_log = []
test_log = []
best_acc = 0
test_acc_log = []
test_f1_log = []

print("--- Begin training ---")
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")

    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, epoch=epoch, scheduler=None, max_epochs=epochs, device=device)
    test_loss, test_acc, test_f1 = test_loop(val_dataloader, model, loss_fn, split="val", device=device)
    train_log.append(train_loss)
    test_log.append(test_loss)
    test_acc_log.append(test_acc)
    test_f1_log.append(test_f1)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            'accuracy': test_acc,
            'f1-score': test_f1
            }, OUTPUT / 'model.pt')
print("Done!")
df = pd.DataFrame({"train_log":train_log, "test_log":test_log, "test_acc_log":test_acc_log})
df.to_csv(OUTPUT / 'logs.csv')
print("Best val acc: ", best_acc)

model = CNN()
model= nn.DataParallel(model)
model.to(device)
model.module.init_weights(next(iter(train_dataloader))[0].to(device))

checkpoint = torch.load(OUTPUT / 'model.pt')
print("Best model from epoch ", checkpoint["epoch"])
model.load_state_dict(checkpoint['model_state_dict'])

loss, accuracy, f1 = test_loop(test_dataloader, model, loss_fn, split="test", device=device)
print("Test acc: ", accuracy)

print_logs(df, epochs=epochs, file_dir=OUTPUT)

model = load_model(CNN, OUTPUT / "model.pt", test_dataloader, device=device, is_parallel=True, init=False)

confusion_matrix = get_conf_matrix(model, test_dataloader, len(names_classes), device=device)

plot_conf_matrix(confusion_matrix, names_classes, OUTPUT)