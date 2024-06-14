Data should be downloaded using the provided scripts:
```
python3 ./src/ml/data/download_SG_data.py
python3 ./src/ml/data/download_GS_data.py
```
or manually such that the folder structure is:

```
data
    gravity_spy
        trainingsetv1d1_metadata.csv
        trainingsetv1d1.h5
    synthetic_glitches
        figshare_image_dataset_np_CHIRPLIKE_test
            test
                CHIRPLIKE
                    0048_CHIRPLIKE_spec_data
                    ...
        ...
        figshare_image_dataset_np_WHISTLELIKE_val
            val
                WHISTLELIKE
                    0039_WHISTLELIKE_spec_data
                    ...
```