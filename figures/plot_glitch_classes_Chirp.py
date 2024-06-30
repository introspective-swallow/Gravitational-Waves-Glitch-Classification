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
classes = ["Blip", "Chirp"]
gls = []
gl_im05 = []
gl_im1 = []
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
        gl_im4.append(np.array(test[gl["gravityspy_id"]]["4.0.png"][:]))

fig, axs = plt.subplots(2,2, figsize=(14*0.8,17*0.8))

axss = axs.flatten()

# Plot the 1s images
for i in range(2):
    axss[i].imshow(gl_im1[i].squeeze())
    axss[i].set_title(gls[i]["label"].replace("_"," "), fontsize=20)
    axss[i].set_xticks([])
    axss[i].set_yticks([])


# Plot the 4s images
for i in range(2):
    axss[2+i].imshow(gl_im4[i].squeeze())
    axss[2+i].set_xticks([])
    axss[2+i].set_yticks([])

axss[0].set_ylabel("1s", fontsize=20)
axss[2].set_ylabel("4s", fontsize=20)

plt.tight_layout()
plt.savefig("/home/gui/OneDrive/Mathematics/TFG/Latex/Figs/" + "chirp-blip.png", transparent=True, bbox_inches='tight')

plt.show()