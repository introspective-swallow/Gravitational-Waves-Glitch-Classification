import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from gwpy.plot import Plot
from gwpy.timeseries import TimeSeries
from gwpy.time import from_gps

DATA = Path("../data/gravity_spy")

dataset_1 = pd.read_csv(DATA / "trainingset_v1d1_metadata.csv")
blip_gps = dataset_1[dataset_1["label"]=="Blip"].iloc[0]["event_time"]
time_window = 512
hdata = TimeSeries.fetch_open_data('H1', int(blip_gps)-time_window, int(blip_gps)+time_window)

#-- Use OUTSEG for small time range
hq2 = hdata.q_transform(frange=(10, 1000), outseg=(blip_gps-0.25,blip_gps+0.25)) 
plot = hq2.plot()

ax = plot.gca()
ax.set_epoch(blip_gps)
ax.set_yscale('log')
ax.set_title('Q-transform')
ax.set_ylabel('Freqüència [Hz]')
ax.set_xlabel('Temps [segons] centrat en ' + from_gps(blip_gps).isoformat(timespec='microseconds') + f' UTC ({blip_gps})')
ax.colorbar(label="Energia normalitzada", clim=[0.1, 10])
