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

hp = hdata.crop(blip_gps-7, blip_gps+7).highpass(20)

white = hp.whiten(4, 2).crop(blip_gps-5, blip_gps+5)

specgram = white.spectrogram2(fftlength=1/32., overlap=7/256.) ** (1/2.)

specgram = specgram.crop_frequencies(20)  # drop everything below highpass

specgram = specgram.ratio('median')

plot = specgram.plot(norm='log', cmap='viridis', yscale='log')
ax = plot.gca()
ax.set_title('Espectrograma')
ax.set_xlim(blip_gps-0.25, blip_gps+0.25)
ax.set_epoch(blip_gps)
ax.colorbar(label='Amplitud relativa', clim=[0.1, 10])

ax.set_ylabel('Freqüència [Hz]')
ax.set_yscale('log')
ax.set_xlabel('Temps [segons] centrat en ' + from_gps(blip_gps).isoformat(timespec='microseconds') + f' UTC ({blip_gps})')

plot.show()
plt.savefig('/home/gui/OneDrive/Mathematics/TFG/Latex/Figs/blip_specgram.png')