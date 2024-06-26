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


Training over simulated glitches dataset:

<img src="figures/ex-sint-glitch-train.png" alt="drawing" width="400"/>

Best models over Gravity Spy dataset

| Resolution | Accuracy | F1-score |
|-----------|-----------|----------|
| 0.5s      | 94.7%     | 93.5%    |
| 1s        | 95.9%     | 92.9%    |
| 2s        | 96.2%     | 90.0%    |
| 4s        | 94.4%     | 91.2%    |


Training over Gravity Spy dataset:

<img src="figures/ex-GS-glitch-train.png" alt="drawing" width="400"/>
