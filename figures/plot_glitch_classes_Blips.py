import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


DATA = Path("./data/gravity_spy")
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

#classes = ["Blip", "Chirp", "Koi_Fish", "Helix", "Low_Frequency_Burst", "Scattered_Light"]
classes = ["Repeating_Blips", "Blip"]
gls = []
gl_im05 = []
gl_im1 = []

for cl in classes:
    gls.append(dataset[dataset["label"]==cl].iloc[0])



with h5py.File(DATASET_PATH, 'r') as file:
    a_group_key = list(file.keys())[0]
     
    # Getting the data
    data = list(file[a_group_key])
    for gl in gls:
        test = file[gl["label"]][gl["sample_type"]]
        gl_im05.append(np.array(test[gl["gravityspy_id"]]["0.5.png"][:]))
        gl_im1.append(np.array(test[gl["gravityspy_id"]]["1.0.png"][:]))

fig, axs = plt.subplots(1,3, figsize=(14*0.8,17*0.8))

axss = axs.flatten()

axss[0].imshow(gl_im1[0].squeeze())
axss[0].set_title(gls[0]["label"].replace("_"," ") + " (1s)", fontsize=20)
axss[0].set_xticks([])
axss[0].set_yticks([])

axss[1].imshow(gl_im05[0].squeeze())
axss[1].set_title(gls[0]["label"].replace("_"," ") + " (0.5s)", fontsize=20)
axss[1].set_xticks([])
axss[1].set_yticks([])

axss[2].imshow(gl_im05[1].squeeze())
axss[2].set_title(gls[1]["label"].replace("_"," ") + " (0.5s)", fontsize=20)
axss[2].set_xticks([])
axss[2].set_yticks([])


#plt.savefig("/home/gui/OneDrive/Mathematics/TFG/Latex/Figs/" + "repeat_blip.png", transparent=True, bbox_inches='tight')
plt.show()

