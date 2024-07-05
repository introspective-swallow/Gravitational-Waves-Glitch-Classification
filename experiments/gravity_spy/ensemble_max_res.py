import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from ml.data.dataset_GS import WaveFormDataset
from ml.data.preprocess_GS import process_image_augment_test
from ml.model.CNN import CNN
from pathlib import Path
from ml.utils.utils import get_device

DATA = Path("./data/gravity_spy")
DATASET_PATH = DATA / "trainingsetv1d0.h5"
METADATA_PATH = DATA / "trainingset_v1d0_metadata.csv"
OUTPUT = Path("./experiments/gravity_spy/exp1")

# Fill with the paths to the models
PATH_CNN05 = ""
PATH_CNN1 = ""
PATH_CNN2 = ""
PATH_CNN4 = ""

device = get_device()

# Resnet18 pretrained adapted to 1 channel
def CNN_pretrained(num_classes=22, PATH="", dummy_dims=(64, 1, 140, 170), device=device, hidden_units=64):
    model = CNN()
    model= nn.DataParallel(model)
    model.to(device)
    model.module.init_weights(torch.randn(dummy_dims).to(device))
    print("Loading ", PATH)
    checkpoint = torch.load(Path(PATH)/"model.pt", map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.module.to(device)
    return model



CNN05 = CNN_pretrained(PATH=PATH_CNN05, dummy_dims=(64,1,128,128), device=device)
CNN1 =  CNN_pretrained(PATH=PATH_CNN1, dummy_dims=(64,1,128,128), device=device)
CNN2 =  CNN_pretrained(PATH=PATH_CNN2, dummy_dims=(64,1,128,128), device=device)
CNN4 =  CNN_pretrained(PATH=PATH_CNN4, dummy_dims=(64,1,128,128), device=device)

dataset05 = WaveFormDataset(DATASET_PATH, METADATA_PATH, "test", res=0.5, transform=process_image_augment_test, device=device)
dataloader05 = DataLoader(dataset05, batch_size=1287)

dataset1 = WaveFormDataset(DATASET_PATH, METADATA_PATH, "test", res=1, transform=process_image_augment_test, device=device)
dataloader1 = DataLoader(dataset1, batch_size=1287)

dataset2 = WaveFormDataset(DATASET_PATH, METADATA_PATH, "test", res=2, transform=process_image_augment_test, device=device)
dataloader2 = DataLoader(dataset2, batch_size=1287)

dataset4 = WaveFormDataset(DATASET_PATH, METADATA_PATH, "test", res=4, transform=process_image_augment_test, device=device)
dataloader4 = DataLoader(dataset4, batch_size=1287)


CNN05.eval()
CNN1.eval()
CNN2.eval()
CNN4.eval()

all_preds_max_all = []
all_preds_max_3 = []

all_labels = []
for (X05, y05),(X1,y1),(X2,y2),(X4,y4) in zip(tqdm(dataloader05),dataloader1,dataloader2,dataloader4):
    outputs05 = CNN05(X05)
    outputs1 = CNN1(X1)
    outputs2 = CNN2(X2)
    outputs4 = CNN4(X4)
    
    output_all = torch.stack([outputs05,outputs1,outputs2,outputs4],axis=0)
    output_all, _ = torch.max(output_all, 0)
    _, preds_max_all = torch.max(output_all, 1)

    all_preds_max_all.extend(preds_max_all.cpu().numpy())

    output_3 = torch.stack([outputs05,outputs1,outputs2],axis=0)
    output_3, _ = torch.max(output_3, 0)
    _, preds_max_3 = torch.max(output_3, 1)

    all_preds_max_3.extend(preds_max_3.cpu().numpy())

    all_labels.extend(y05.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds_max_all)
f1 = f1_score(all_labels, all_preds_max_all, average='macro')

print(f"All metrics: \n Accuracy: {(100*accuracy):>0.2f}%, F1-score: {(100*f1):>0.2f}%")

accuracy = accuracy_score(all_labels, all_preds_max_3)
f1 = f1_score(all_labels, all_preds_max_3, average='macro')

print(f"3 metrics: \n Accuracy: {(100*accuracy):>0.2f}%, F1-score: {(100*f1):>0.2f}%")
