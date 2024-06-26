import matplotlib.pyplot as plt
import numpy as np
import math

t = np.linspace(0,5.5, 1000)
freq = 1
periode = 1 / freq
x = np.sin(2* math.pi*freq*t)
sr = 13*freq
T = 1 / sr
td = np.arange(0,5.5, T)
xd = np.sin(2* math.pi*freq*td)
fig, ax = plt.subplots()
ax.scatter(td, xd, color="black", s=30)
ax.plot(t, x, '--')
ax.set_yticks([])
ax.set_xticks([])
ax.axhline(y=0, color='black')
ax.grid(True, which='both')

plt.show()