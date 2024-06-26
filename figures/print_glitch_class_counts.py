from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import h5py 
import numpy as np
from tqdm import tqdm

DATA = Path("../data/gravity_spy")
DATASET_PATH = DATA / "trainingsetv1d0.h5"
METADATA_PATH = DATA / "trainingset_v1d0_metadata.csv"

with h5py.File(DATASET_PATH, 'r') as f:
    names_classes = list(f.keys())
    num_classes = len(names_classes)
    splits = list(f[names_classes[0]].keys())
    file_name_test = list(f[names_classes[0]][splits[0]].keys())[0]
    test = f[names_classes[0]][splits[0]][file_name_test]
    resolutions = list(test)
dataset = pd.read_csv(METADATA_PATH)
class_counts = [sum(dataset["label"]==glitch) for glitch in names_classes]

datas05 = {"train":[],
           "validation":[],
           "test":[]}

datas10 = []
datas20 = {"train":[],
           "validation":[],
           "test":[]}
datas40 = []

labels = {"train":[],
          "validation":[],
          "test":[]}

data = datas20

with h5py.File(DATASET_PATH, 'r') as f:
    for i in tqdm(range(len(dataset))):
        id = dataset["gravityspy_id"][i]
        split = dataset["sample_type"][i]
        label = dataset["label"][i]

        labels[split].append(names_classes.index(label))
        data[split].append(np.array(f[label][split][id][resolutions[0]]))

X_train = np.array(data["train"])
y_train = np.array(labels["train"])

X_val = np.array(data["validation"])
y_val = np.array(labels["validation"])

X_test = np.array(data["test"])
y_test = np.array(labels["test"])

test_i = 29
print(X_train.shape)
test_im = X_train[test_i,0,:,:]
test_label = y_train[test_i]
height, width = test_im.squeeze().shape
plt.imshow(test_im.squeeze(), origin="lower")
plt.title(names_classes[test_label])
train_class_counts = [sum(y_train==glitch) for glitch in range(len(names_classes))] 
val_class_counts = [sum(y_val==glitch) for glitch in range(len(names_classes))] 
test_class_counts = [sum(y_test==glitch) for glitch in range(len(names_classes))] 

# Create the LaTeX table
def generate_latex_table(classes, train_samples, val_samples, test_samples):
    table = r"\begin{table}[h!]" + "\n"
    table += r"\centering" + "\n"
    table += r"\begin{tabular}{|l|c|c|c|}" + "\n"
    table += r"\hline" + "\n"
    table += r"Class & Train & Validation & Test \\" + "\n"
    table += r"\hline" + "\n"

    for cls, train, val, test in zip(classes, train_samples, val_samples, test_samples):
        table += f"{cls} & {train} & {val} & {test} \\\\" + "\n"
        table += r"\hline" + "\n"

    table += r"\end{tabular}" + "\n"
    table += r"\caption{Sample distribution across Train, Validation, and Test sets}" + "\n"
    table += r"\label{table:sample_distribution}" + "\n"
    table += r"\end{table}"

    return table

# Generate and print the LaTeX table
latex_table = generate_latex_table(names_classes, train_class_counts, val_class_counts, test_class_counts)
print(latex_table)
