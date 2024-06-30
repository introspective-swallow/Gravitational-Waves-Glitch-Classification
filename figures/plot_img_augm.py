import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ml.data.preprocess_GS import process_image_augment_train, process_image_augment_test
from ml.utils.utils import seed_everything

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

classes = ["Blip", "Chirp", "Koi_Fish", "Helix", "Low_Frequency_Burst", "Scattered_Light"]

test_img_row = dataset[dataset["label"]==classes[0]].iloc[0]


with h5py.File(DATASET_PATH, 'r') as file:
    a_group_key = list(file.keys())[0]
     
    # Getting the data
    data = list(file[a_group_key])
    test = file[test_img_row["label"]][test_img_row["sample_type"]]
    test_img = np.array(test[test_img_row["gravityspy_id"]]["1.0.png"][:])

seed_everything(42)
test_img_augm1 = process_image_augment_train(test_img)
test_img_augm2 = process_image_augment_test(test_img)
test_img_augm3 = process_image_augment_train(test_img)

fig, axs = plt.subplots(2, 2)

axs_flat = axs.flatten()

axs_flat[0].imshow(test_img.squeeze())
axs_flat[0].set_xticks([])
axs_flat[0].set_yticks([])

for i, img in enumerate([test_img_augm1, test_img_augm2, test_img_augm3]):
    axs_flat[i+1].imshow(img.squeeze())
    axs_flat[i+1].set_xticks([])
    axs_flat[i+1].set_yticks([])

#plt.tight_layout()
#plt.savefig("/home/gui/OneDrive/Mathematics/TFG/Latex/Figs/" + "img_augm.png", transparent=True, bbox_inches='tight')

plt.show()