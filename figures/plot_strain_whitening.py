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

plot = hdata.plot()

ax = plot.gca()
ax.set_epoch(blip_gps)
ax.set_xlim(blip_gps-0.25, blip_gps+0.25)

hwdata = hdata.whiten()
plot = hwdata.plot()

ax = plot.gca()
ax.set_epoch(blip_gps)
ax.set_xlim(blip_gps-0.25, blip_gps+0.25)

plot = Plot(hdata, hwdata, figsize=(12, 6), separate=True, sharex=True)
ax = plot.gca()
ax.set_epoch(blip_gps)
ax.set_xlim(blip_gps-0.25, blip_gps+0.25)
ax.set_xlabel('Temps [segons] centrat en ' + from_gps(blip_gps).isoformat(timespec='microseconds') + f' UTC ({blip_gps})')
axs = plot.get_axes()
axs[0].set_ylabel('Deformació')
axs[1].set_ylabel('Deformació blanquejada')
plt.savefig('/home/gui/OneDrive/Mathematics/TFG/Latex/Figs/blip_strain.png')