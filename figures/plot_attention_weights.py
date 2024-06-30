import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml.data.preprocess_GS import process_parallel_image_augment_test_post
from ml.data.dataset_GS import WaveFormDatasetParallelPost
from ml.model.CNN_attention_VGG import CNN_attention
from ml.utils.utils import get_device, get_metadata
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from pathlib import Path


DATA = Path("./data/gravity_spy")
DATASET_PATH = DATA / "trainingsetv1d0.h5"
METADATA_PATH = DATA / "trainingset_v1d0_metadata.csv"

# Fill in the path to the model
PATH_CNN_attention_VGG = ""

def load_model(model_constr, saved_model_path, dummy_dims=((64,4,140,170)), params={},
    is_parallel=True, init=True, device="cuda"):
    model = model_constr(**params)
    if is_parallel:
        model= nn.DataParallel(model)
    model.to(device)
    if init:
        model.module.init_weights(torch.randn(dummy_dims).to(device))

    checkpoint = torch.load(saved_model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.module.to(device)
    acc = checkpoint['accuracy']
    f1_score = checkpoint['f1-score']
    epoch = checkpoint['epoch']
    return model, acc, f1_score, epoch

device = get_device()
params = {"dropout":0.3, "arch":((3, 64), (3, 64), (3, 256), (3, 256)), "latent_vectors":128, "hidden_units":128}

model, acc, f1, _ = load_model(CNN_attention, PATH_CNN_attention_VGG, (64, 4, 140, 170),
                   device="cuda",params=params)
print(acc, f1)

dataset = WaveFormDatasetParallelPost(DATASET_PATH, METADATA_PATH, "test",
                                      transform=process_parallel_image_augment_test_post, device="cuda")
dataloader = DataLoader(dataset, batch_size=32)

model.eval()
preds = []

labels = []
with torch.no_grad():
    for (X, y) in tqdm(dataloader):
        X = X.to(device)
        y = y.to(device)   
        output = model(X)
        _, pred = torch.max(output, 1)
        preds.extend(pred.cpu().numpy())
        labels.extend(y.cpu().numpy())

accuracy = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average='macro')

print(f"All metrics: \n Accuracy: {(100*accuracy):>0.2f}%, F1-score: {(100*f1):>0.2f}%")


model.eval()
preds = []
weights1 = []
weights2 = []
weights3 = []
weights4 = []

with torch.no_grad():
    for (X, y) in tqdm(dataloader):
        X = X.to(device)
        y = y.to(device)  
        h1 = model.net[0].h1(X[:,0,:,:].unsqueeze(1))
        h2 = model.net[0].h2(X[:,1,:,:].unsqueeze(1))
        h3 = model.net[0].h3(X[:,2,:,:].unsqueeze(1))
        h4 = model.net[0].h4(X[:,3,:,:].unsqueeze(1))

        multi = torch.stack((h1, h2, h3, h4), axis=1)

        attention = model.net[0].attention

        weights1.extend(attention.attentionWeight(multi[:,0,:]).cpu().detach().numpy())
        weights2.extend(attention.attentionWeight(multi[:,1,:]).cpu().detach().numpy())
        weights3.extend(attention.attentionWeight(multi[:,2,:]).cpu().detach().numpy())
        weights4.extend(attention.attentionWeight(multi[:,3,:]).cpu().detach().numpy())


w1 = []
w2 = []
w3 = []
w4 = []
for i in range(len(weights1)):
    w1.extend(weights1[i])
    w2.extend(weights2[i])
    w3.extend(weights3[i])
    w4.extend(weights4[i])
weights = np.stack([w1,w2,w3,w4],axis=1)
df = pd.DataFrame(weights)
df.to_csv("/kaggle/working/weights.csv")

labels = np.array(labels)
weights_per_class = []
for i in range(22):
    weights_per_class.append(weights[labels==i].mean(axis=0))
meta = get_metadata(DATASET_PATH, METADATA_PATH)
names = meta["names_classes"]
w = np.array(weights_per_class) 

x = [n[:2] for n in names]
plt.bar(x, w[:,0], color='r')
plt.bar(x, w[:,1], bottom=w[:,0], color='b')
plt.bar(x, w[:,2], bottom=w[:,0]+w[:,1], color='y')
plt.bar(x, w[:,3], bottom=w[:,0]+w[:,1]+w[:,2], color='g')

# Add legend
plt.legend(["0.5s", "1s", "2s", "4s"], title="Resolutions")

# Add x axis title
plt.xlabel("Glitch")
plt.ylabel("Pesos d'atenció")

plt.show()