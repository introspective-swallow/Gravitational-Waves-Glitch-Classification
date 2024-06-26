import os
import pandas as pd
import torch
import re

DATA_PATH = "/home/gui/OneDrive/Mathematics/TFG/Models/v0/kaggle/results/"

# Get all logs from the models, the columns are:
# Model	Input Data Type	Loss Function	Accuracy	F1-score	Avg Loss

logs = pd.read_csv(DATA_PATH + "all_logs.csv")

# show best models by f1 score
print("Best by f1\n", logs.sort_values("F1-score", ascending=False).head(3))

# show best models by accuracy
print("Best by acc\n", logs.sort_values("Accuracy", ascending=False).head(3))

# Plot models f1 by loss type, model type and merged/res input data
import matplotlib.pyplot as plt
import seaborn as sns

# Plot models f1 by loss type, model type and merged/res input data

fig, ax = plt.subplots(1, 3, figsize=(20, 5))
sns.boxplot(data=logs, x="Loss Function", y="F1-score", ax=ax[0])

sns.boxplot(data=logs, x="Model", y="F1-score", ax=ax[1])

sns.boxplot(data=logs, x="Input Data Type", y="F1-score", ax=ax[2])

plt.show()

# Plot models acc by loss type, model type and merged/res input data

fig, ax = plt.subplots(1, 3, figsize=(20, 5))

sns.boxplot(data=logs, x="Loss Function", y="Accuracy", ax=ax[0])

sns.boxplot(data=logs, x="Model", y="Accuracy", ax=ax[1])

sns.boxplot(data=logs, x="Input Data Type", y="Accuracy", ax=ax[2])

plt.show()

# Also make one unique plot of the f1 by loss type, model type and merged/res input data

fig, ax = plt.subplots(1, 1, figsize=(20, 5))
sns.boxplot(data=logs, x="Loss Function", y="F1-score", hue="Model", ax=ax)

plt.show()

# Also make one unique plot of the acc by loss type, model type and merged/res input data

fig, ax = plt.subplots(1, 1, figsize=(20, 5))
sns.boxplot(data=logs, x="Loss Function", y="Accuracy", hue="Model", ax=ax)

plt.show()