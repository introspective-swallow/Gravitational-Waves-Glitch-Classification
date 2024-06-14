from urllib.request import urlretrieve
from pathlib import Path

GS_DATA = Path("./data/gravity_spy")

if not os.path.exists(GS_DATA):
    os.makedirs(GS_DATA)

# or second version https://zenodo.org/records/1486046

try:
    urlretrieve("https://zenodo.org/records/1476156/files/trainingset_v1d0_metadata.csv?download=1", GS_DATA/"trainingset_v1d0_metadata.csv")
except:
    print("Error downloading Gravity Spy metadata file.")

try:
    urlretrieve("https://zenodo.org/records/1476156/files/trainingsetv1d0.h5?download=1", GS_DATA/"trainingsetv1d0.h5")
except:
    print("Error downloading Gravity Spy images file.")

