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

from ml.data.preprocess_GS import process_image_augment_train, process_image_augment_test
from ml.model.CNN import CNN3
from ml.data.dataset_GS import WaveFormDatasetPost
from ml.training.trainer import Trainer
from ml.utils.utils import get_device, next_exp

DATA = Path("./data/gravity_spy")
DATASET_PATH = DATA / "trainingsetv1d0.h5"
METADATA_PATH = DATA / "trainingset_v1d0_metadata.csv"
OUTPUT = Path(".models/res05") 

trainer = Trainer(DATASET_PATH=DATASET_PATH, METADATA_PATH=METADATA_PATH, file_dir=next_exp(OUTPUT), epochs=50, 
                batch_size=64, weighted_loss=False, loss="baseline", lr = 0.001,
                model_class=CNN3, res=0.5, dataset=WaveFormDatasetPost,
                scheduler_active=True, patience=2, process_train=process_image_augment_train,
                  process_test=process_image_augment_test, img_augmentation=True, init_model=True, use_amp=True)
trainer.train_test()
