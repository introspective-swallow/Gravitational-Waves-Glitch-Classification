import matplotlib.pyplot as plt
import numpy as np
import math

t1 = 0
t2 = 8
t = np.linspace(t1,t2, 1000)

freq = 1/8
periode = 1 / freq
x = np.sin(2* math.pi*freq*t)

freq2 = -7 / 8
x2 = np.sin(2* math.pi*freq2*t)

sr = 1
T = 1 / sr

td = np.arange(t1,t2+1, T)
xd = np.sin(2* math.pi*freq*td)
fig, ax = plt.subplots()
ax.scatter(td, xd, color="black", s=30)
ax.plot(t, x, '--')
ax.plot(t, x2, '--', color="red")
ax.set_yticks([])
ax.set_xticks([])
ax.axhline(y=0, color='black')
ax.grid(True, which='both')

plt.show()