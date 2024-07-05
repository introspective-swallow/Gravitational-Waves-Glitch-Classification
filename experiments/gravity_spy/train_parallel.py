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

from ml.data.preprocess_GS import process_parallel_image_augment_train_post, process_parallel_image_augment_test_post
from ml.model.CNN_parallel import CNN_parallel
from ml.data.dataset_GS import WaveFormDatasetParallelPost, get_metadata
from ml.training.trainer import Trainer
from ml.utils.utils import get_device, next_exp

DATA = Path("./data/gravity_spy")
DATASET_PATH = DATA / "trainingsetv1d0.h5"
METADATA_PATH = DATA / "trainingset_v1d0_metadata.csv"
OUTPUT = Path("./models/parallel/")

params = {"dropout":0.25, "arch1":((5, 64, 2), (5,128,2)), "arch2":((5,128,2),(5, 128,0),(5,256,0))}

trainer = Trainer(DATASET_PATH=DATASET_PATH, METADATA_PATH=METADATA_PATH, file_dir=next_exp(OUTPUT), 
                 loss="baseline", epochs=50, batch_size=64, lr=0.001,
                 weighted_loss=True, model_class=CNN_parallel, dataset=WaveFormDatasetParallelPost,
                 model_params= params, device=get_device(), scheduler_active=True, patience=2, init_model=True,
                 process_train=process_parallel_image_augment_train_post,
                 process_test=process_parallel_image_augment_test_post, img_augmentation=True)

trainer.train_test()