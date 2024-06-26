import pandas as pd
import numpy as np
import os
import random
import time
from tqdm import tqdm
from pathlib import Path

from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

from ml.data.preprocess_SG import normalize_percentile, process_image
from ml.data.dataset_SG import get_raw_dataset, get_dataset
from ml.utils.utils import seed_everything

seed_everything(0)

DATA = Path("./data/synthetic_glitches")
OUTPUT = Path("./experiments/synthetic_glitches")

train_annotations, train_files, val_annotations, val_files, test_annotations, test_files = get_raw_dataset(DATA)

X_train, y_train = get_dataset(train_annotations, train_files)
X_test, y_test = get_dataset(test_annotations, test_files)

# Make pipeline to learn scaling while training
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('RF', RandomForestClassifier(random_state=0, verbose=1))
])

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, pred)

f1 = f1_score(y_test, pred, average='weighted')

print("Acc: ", acc, "F1: ", f1)

