import os
import pandas as pd
import torch
import re

DATA_PATH = "/home/gui/OneDrive/Mathematics/TFG/Models/v0/kaggle/results/"

logs = []

for dir in os.listdir(DATA_PATH):
    if not os.path.exists(os.path.join(DATA_PATH, dir, "model.pt")):
        continue
    checkpoint = torch.load(os.path.join(DATA_PATH, dir, "model.pt"), map_location=torch.device('cpu'))
    acc, f1 = checkpoint["accuracy"], checkpoint["f1-score"]
    model_name = dir
    results = {"model_name": model_name, "acc": acc, "f1": f1}
    logs.append(results)

logs = pd.DataFrame(logs)

# To check if it is resnet18 or convnext, check if the model is contained in the name

logs["model_type"] = logs["model_name"].apply(lambda x: "resnet18" if "resnet18" in x else "convnext")

# To check the loss type, check if the name contains "lossbaseline" or "lossfocal"

logs["loss_type"] = logs["model_name"].apply(lambda x: "baseline" if "lossbaseline" in x else "focal")

# To check if the input data is merged or not, check if the name contains "merged"
# if it doesn't get the resolution by matching with a regex the number in res0.5, res1, res2, res4 

logs["input_data"] = logs["model_name"].apply(lambda x: "merged" if "merged" in x else f"{re.search(r'res\d+', x).group(0)}")

logs = logs.sort_values("f1", ascending=False)
logs.to_csv(DATA_PATH + "all_logs.csv")
