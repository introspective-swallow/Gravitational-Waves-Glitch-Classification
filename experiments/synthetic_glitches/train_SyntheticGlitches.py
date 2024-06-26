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

from ml.data.preprocess_SG import normalize_percentile, process_image
from ml.data.dataset_SG import get_raw_dataset, get_dataset
from ml.utils.utils import seed_everything
from ml.model.CNN import CNN3 as CNN
from ml.data.dataset_SG import WaveFormDatasetFast as WaveFormDataset, get_raw_dataset
from ml.training.loops import train_loop, test_loop
from ml.utils.utils import seed_everything, print_logs, load_model, get_conf_matrix, plot_conf_matrix, get_device
from ml.losses.focal_loss import FocalLoss

# Set seed
seed_everything(0)

device = get_device()
names_classes = ['GAUSS','CHIRPLIKE','RD','SCATTEREDLIKE','SG','WHISTLELIKE','NOISE']
DATA = Path("./data/synthetic_glitches")
OUTPUT = Path("./experiments/synthetic_glitches")

# Hyperparameters
file_dir = OUTPUT / "CNN3"
batch_size = 20
lr = 0.0008
weight_decay = 0
label_smoothing = 0
epochs = 20
scheduler_active = True
patience = 2
loss = "baseline"
model_params = {}
use_amp = True

# Get data

train_annotations, train_files, val_annotations, val_files, test_annotations, test_files = get_raw_dataset(DATA)

training_data = WaveFormDataset(train_annotations, train_files, transform=process_image)
val_data = WaveFormDataset(val_annotations, val_files, split="val", transform=process_image)
test_data = WaveFormDataset(test_annotations, test_files, split="test", transform=process_image)

seed_everything(0)

if not os.path.exists(file_dir):
    os.makedirs(file_dir)

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

if loss=="baseline":
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
elif loss=="focal":
    loss_fn = FocalLoss()
#optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer = torch.optim.AdamW(model.parameters(), lr= lr, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr= lr, weight_decay=weight_decay)

if scheduler_active:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True, patience=patience, min_lr=1e-8)
else:
    scheduler=None

params = {"batch_size":[batch_size], "lr":[lr], "weight_decay":[label_smoothing], "epochs":[epochs], "scheduler_active": [scheduler_active], "patience":[patience],
                    "optimizer":[type(optimizer).__name__], "loss":[loss], "use_amp":[use_amp]}
params.update( model_params)

df = pd.DataFrame.from_dict(params)
df.to_csv(file_dir / "params.csv")

# Train model
epochs = epochs

train_log = []
test_log = []
best_metric = 0
test_acc_log = []
test_f1_log = []

print("--- Begin training ---")
for epoch in range(epochs):
    train_loss = train_loop(train_dataloader,  model, loss_fn, optimizer, epoch=epoch, scheduler=scheduler, max_epochs=epochs, device= device, use_amp= use_amp)
    test_loss, test_acc, test_f1 = test_loop(val_dataloader,  model, loss_fn, split="val", device= device)
    train_log.append(train_loss)
    test_log.append(test_loss)
    test_acc_log.append(test_acc)
    test_f1_log.append(test_f1)
    if test_f1 > best_metric:
        best_metric = test_f1
        torch.save({
            'epoch': epoch,
            'model_state_dict':  model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            'accuracy': test_acc,
            'f1-score': test_f1
            },  file_dir / 'model.pt')
print("Done!")
df = pd.DataFrame({"train_log":train_log, "test_log":test_log, "test_acc_log":test_acc_log, "test_f1_log":test_f1_log})
df.to_csv( file_dir/"logs.csv")
print_logs(df, epochs=epochs, file_dir= file_dir)

model, best_acc, best_f1 = load_model(CNN,  file_dir / "model.pt", test_dataloader, params= model_params, device= device, is_parallel=True, init= True)
print(f"Best val acc: {(100*best_acc):>0.2f}%, Best val f1: {(100*best_f1):>0.2f}%")
loss, accuracy, f1 = test_loop(test_dataloader,  model, loss_fn, split="test", device= device)
df_test = pd.DataFrame({"accuracy":[accuracy], "f1":[f1]})
df_test.to_csv( file_dir/"metrics.csv")

confusion_matrix = get_conf_matrix( model, test_dataloader, len( names_classes), device= device)
plot_conf_matrix(confusion_matrix,  names_classes,  file_dir)  
