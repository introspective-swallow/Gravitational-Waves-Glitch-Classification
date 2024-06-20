import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    
def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device.")
    return device

def load_model(model_constr, saved_model_path, dataloader, params={}, is_parallel=True, init=True, device="cpu"):
    model = model_constr(**params)
    if is_parallel:
        model= nn.DataParallel(model)
    model.to(device)
    if init:
        model.module.init_weights(next(iter(dataloader))[0].to(device))

    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    acc = checkpoint['accuracy']
    f1_score = checkpoint['f1-score']
    return model, acc, f1_score
    
def get_conf_matrix(model, loader, num_classes, device="cpu"):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(loader):
            inputs = inputs.to(device)
            classest = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classest.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix

def print_logs(df, epochs=50, file_dir=None):
    fig, ax = plt.subplots()
    ax.plot(df["train_log"], label="cost entrenament", color="blue",alpha=0.5, linestyle="--")
    ax.plot(df["test_log"], label="cost evaluació", color="red", alpha=0.7)
    ax.plot(df["test_acc_log"], label="precissió evaluació", color="green", alpha=0.7)
    ax.axhline(1, color="black", linestyle="--")
    ax.set_xticks([0, *list(range(49, epochs, 50))], [1, *list(range(50,epochs+1, 50))])
    ax.set_ylim([0,1.2])
    ax.legend()
    plt.show()
    if file_dir:
        plt.savefig(file_dir / "logs.png")

def plot_conf_matrix(conf_matrix, names_classes, file_dir=None):
    fig, ax = plt.subplots(figsize=(15,15))
    cmap = sns.color_palette("Greens", as_cmap=True)
    ax = sns.heatmap(np.array(conf_matrix/conf_matrix.sum(axis=1, keepdim=True)), annot=True, cmap=cmap, cbar=False, xticklabels=names_classes, yticklabels=names_classes)
    ax.set(xlabel="Prediccions", ylabel="Vertaderes")
    plt.tight_layout()
    plt.show()
    if file_dir:
        fig.savefig(file_dir / "conf_matrix.png")