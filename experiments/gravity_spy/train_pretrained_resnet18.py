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

from ml.data.preprocess_GS import process_image
from ml.model.ResNet18 import ResNet18_pretrained3Channel
from ml.data.dataset_GS import WaveFormDatasetMerged3Channel, get_metadata
from ml.training.loops import train_loop, test_loop
from ml.utils.utils import load_model, get_conf_matrix, print_logs, plot_conf_matrix, get_device
from ml.training.trainer import Trainer

DATA = Path("./data/gravity_spy")
DATASET_PATH = DATA / "trainingsetv1d0.h5"
METADATA_PATH = DATA / "trainingset_v1d0_metadata.csv"
OUTPUT = Path("/kaggle/working/transfer_learning3channel")

trainer = Trainer(DATASET_PATH, METADATA_PATH, OUTPUT, loss="focal", epochs=40, batch_size=64, lr=0.01,
                  weighted_loss=True, model_class=ResNet18_pretrained3Channel, dataset=WaveFormDatasetMerged3Channel,
                  model_params={"num_classes": 22}, device=get_device(), scheduler_active=True, init_model=False)

trainer.train_test()

# unfreeze all layers
for param in trainer.model.parameters():
    param.requires_grad = True

trainer.new_parameters(lr=1e-5, epochs=40)
trainer.keep_training()