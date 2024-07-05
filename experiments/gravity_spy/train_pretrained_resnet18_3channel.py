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
from ml.data.dataset_GS import WaveFormDatasetMerged3ChannelPost, get_metadata
from ml.data.preprocess_GS import process_merged_image_augment_train_post3channel128, process_merged_image_augment_test_post3channel128
from ml.training.loops import train_loop, test_loop
from ml.utils.utils import load_model, get_conf_matrix, print_logs, plot_conf_matrix, get_device
from ml.training.trainer import Trainer

DATA = Path("./data/gravity_spy")
DATASET_PATH = DATA / "trainingsetv1d0.h5"
METADATA_PATH = DATA / "trainingset_v1d0_metadata.csv"
OUTPUT = Path("./models/transfer_learning3channel")

trainer = Trainer(DATASET_PATH, METADATA_PATH, OUTPUT, loss="baseline", epochs=50, batch_size=64, lr=0.001,
                  weighted_loss=True, model_class=ResNet18_pretrained3Channel, dataset=WaveFormDatasetMerged3ChannelPost,
                  model_params={"num_classes": 22}, device=get_device(), scheduler_active=True, init_model=False,
                  process_train=process_merged_image_augment_train_post3channel128,
                  process_test=process_merged_image_augment_test_post3channel128, img_augmentation=True)

trainer.train_test()

# unfreeze all layers
for param in trainer.model.parameters():
    param.requires_grad = True

trainer.new_parameters(lr=1e-5, epochs=40, file_dir=Path("./models/transfer_learning3channel/fine-tune"))
trainer.train_test(keep_training=True)