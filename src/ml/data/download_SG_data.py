from urllib.request import urlretrieve
from pathlib import Path
import os, zipfile

SG_DATA = Path("./data/synthetic_glitches")

if not os.path.exists(SG_DATA):
    os.makedirs(SG_DATA)

try:
    urlretrieve("https://figshare.com/ndownloader/articles/7166210/versions/1", SG_DATA/"7166210.zip")
except:
    print("Error downloading synthetic glitches compressed file.")

try:
    with zipfile.ZipFile(SG_DATA/"7166210.zip", 'r') as zip_ref:
        zip_ref.extractall(SG_DATA)
    os.remove(SG_DATA/"7166210.zip")
except:
    print("Error extracting synthetic glitches compressed file.")
