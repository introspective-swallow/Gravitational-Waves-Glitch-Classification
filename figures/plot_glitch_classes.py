import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

classes = ["Blip", "Chirp", "Koi_Fish", "Helix", "Low_Frequency_Burst", "Scattered_Light"]
gls = []
gl_im = []
for cl in classes:
    gls.append(dataset[dataset["label"]==cl].iloc[0])



with h5py.File(DATASET_PATH, 'r') as file:
    print("Keys: %s" % file.keys())
    a_group_key = list(file.keys())[0]
     
    # Getting the data
    data = list(file[a_group_key])
    for gl in gls:
        test = file[gl["label"]][gl["sample_type"]]
        gl_im.append(np.array(test[gl["gravityspy_id"]]["1.0.png"][:]))

fig, axs = plt.subplots(3,2, figsize=(14*0.8,17*0.8))

axss = axs.flatten()
for i, gl in enumerate(gl_im):
    axss[i].imshow(gl.squeeze())
    axss[i].set_title(gls[i]["label"].replace("_"," "), fontsize=20)
    axss[i].set_xticks([])
    axss[i].set_yticks([])

plt.tight_layout()
plt.show()