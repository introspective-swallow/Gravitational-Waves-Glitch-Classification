# Gravitational-Waves-Glitch-Detection
Classification of glitches in Gravitational Waves data from:

-  [Images of simulated glitches in Numpy format](https://doi.org/10.6084/m9.figshare.7166210.v1). Razzano, Massimiliano (2018)

- [Machine learning for Gravity Spy: Glitch classification and dataset](https://doi.org/10.5281/zenodo.1476156). Bahaadini, Sara1; Noroozi, Vahid; Rohani, Neda1; Coughlin, Scott; Zevin, Michael; Smith, Joshua; Kalogera, Vicky; Aggelos, Katsaggelos (October 31, 2018)

## Set up
Install dependencies
```
pip install --upgrade build
python -m build
```

To get the synthetic glitches dataset in the project structure, run:
```
python3 ./src/ml/data/download_SG_data.py
```
To get the Gravity Spy dataset:
```
python3 ./src/ml/data/download_GS_data.py
```

## Models

Best models over simulated glitches:

| Model  | Precision | Precision (10 models)       |
|--------|-----------|-----------------------------|
| 3xCNN  | 99.90%    | 99.42 ± 0.64%               |
| 1xCNN  | 97.3%     | -                           |
| RF     | 99.44%    | -                           |


Best of each model over Gravity Spy dataset:

| Model | Accuracy | F1-score |
|-----------|-----------|----------|
| 3xCNN 0.5s      | 94.7%     | 93.5%    |
| 3xCNN 1s        | 95.9%     | 92.9%    |
| 3xCNN 2s        | 96.2%     | 90.0%    |
| 3xCNN 4s        | 94.4%     | 91.2%    |
| 3xCNN ensemble 0.5s + 1s + 2s |96,81 %|95,31 %|
| CNN middle fusion | 97,44 % | 96,52 % |
| CNN initial fusion | 97,59 % | 96,37 % |
| CNN attention | 95,37 % | 91,02 % |
| CNN attention VGG | 96,97 % | 95,33 % |
| ResNet18  | 98,06% | 97,59 % |
| ResNet18 pretrained | 98.06% | 97.68% |
| ResNet50 pretrained | 97.70% | 96.01% |
| vit_b_16 |97.31% | 95.8%|
