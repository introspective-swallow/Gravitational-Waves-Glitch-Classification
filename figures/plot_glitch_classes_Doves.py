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

print(names_classes)
#classes = ["Blip", "Chirp", "Koi_Fish", "Helix", "Low_Frequency_Burst", "Scattered_Light"]
#classes = ["Chirp", "Blip", "Light_Modulation", "Repeating_Blips", "Scratchy", "None_of_the_Above"]
classes = ["Paired_Doves"]
gls = []
gl_im05 = []
gl_im1 = []
gl_im2 = []
gl_im4 = []

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
        gl_im2.append(np.array(test[gl["gravityspy_id"]]["2.0.png"][:]))
        gl_im4.append(np.array(test[gl["gravityspy_id"]]["4.0.png"][:]))

fig, axs = plt.subplots(1,2, figsize=(14*0.8,17*0.8))

axss = axs.flatten()

axss[0].imshow(gl_im2[0].squeeze())
axss[0].set_title(gls[0]["label"].replace("_"," "), fontsize=20)
axss[0].set_xticks([])
axss[0].set_yticks([])

axss[1].imshow(gl_im4[0].squeeze())
axss[1].set_title(gls[0]["label"].replace("_"," "), fontsize=20)
axss[1].set_xticks([])
axss[1].set_yticks([])

axss[0].set_ylabel("2s", fontsize=20)
axss[1].set_ylabel("4s", fontsize=20)

plt.tight_layout()
plt.savefig("/home/gui/OneDrive/Mathematics/TFG/Latex/Figs/" + "doves.png", transparent=True, bbox_inches='tight')

plt.show()