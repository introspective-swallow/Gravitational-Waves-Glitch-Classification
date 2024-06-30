import os
import pandas as pd
import torch
import re
from pathlib import Path

ROOT = Path("/home/gui/OneDrive/Mathematics/TFG/Models/GS/v0")
PATHS = [ROOT / name for name in ["multi-attention", "resX", "transfer_learning", "resXz", "resnet"]]



logs = []
dir_list = []

# Get all the directories in all data path
for DATA_PATH in PATHS:
    dir_list = os.listdir(DATA_PATH)

    for dir in dir_list:
        dir_path = DATA_PATH / dir
        if not os.path.isdir(dir_path):
            continue
        for subdir in os.listdir(os.path.join(DATA_PATH, dir)):
            model_path = DATA_PATH / dir / subdir / "model.pt"
            metrics_path = DATA_PATH / dir / subdir / "metrics.csv"
            if not os.path.exists(metrics_path):
                continue
            metrics = pd.read_csv(metrics_path)
            acc = metrics["accuracy"].values[0]
            f1 = metrics["f1"].values[0]
            model_name = dir
            results = {"model_name": model_name, "acc": acc, "f1": f1}
            try:
                params = pd.read_csv(DATA_PATH / dir / subdir / "params.csv")
            except:
                print(f"No info on parameters of model in {DATA_PATH / dir / subdir}")
                params = pd.DataFrame({"num_classes": [22]})
            # if there is no img_augmentation column, add it and set it to False
            if "img_augmentation" not in params.columns:
                params["img_augmentation"] = "consultar"
            results.update(params.iloc[0].to_dict())
            # Add the file path to the results
            results["file_path"] = model_path
            logs.append(results)

logs = pd.DataFrame(logs)
logs = logs.drop(columns=["Unnamed: 0"])
logs = logs.sort_values("f1", ascending=False)
# remove duplicate rows (all columns) and print their file paths
duplicates = logs[logs.duplicated()]
duplicates["file_path"].apply(lambda x: print(x))
logs = logs.drop_duplicates()
logs.to_csv(ROOT / "all_logs.csv")
