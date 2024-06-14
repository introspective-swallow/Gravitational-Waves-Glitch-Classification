import matplotlib.pyplot as plt
import numpy as np
import math

N = 1000

n = np.arange(0,N)
windows = {}

# Hann window
windows["Finestra de Hann"] = np.sin(math.pi * n / N)**2

# Tukey window
def tukey(N, alpha=0.5):
    step1 = alpha * N / 2
    step2 = N - step1

    n1 = np.arange(0,step1)
    n3 = n1[::-1]

    x1 = 0.5*(1-np.cos(2 * math.pi * n1 / (alpha*N)))
    x2 = np.ones(math.ceil(step2) - math.floor(step1))
    x3 = 0.5*(1-np.cos(2 * math.pi * n3 / (alpha*N)))

    x = np.concatenate((x1,x2,x3),axis=0)
    return x


windows["Finestra de Tukey"] = tukey(N)

fig, ax = plt.subplots(nrows=1,ncols=2)

ax_f = ax.flatten()

for i, key in enumerate(windows):
    ax = ax_f[i]
    x = windows[key]
    ax.plot(n, x, '-', lw=1, color="#2c97de")
    d = np.zeros(len(n))
    ax.fill_between(n, x, where=x>=d, interpolate=True, color="#2c97de")

    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(True, which='both')
    ax.set_ylim([0,1.05])
    ax.set_title(key, fontsize=15)

plt.savefig('/home/gui/OneDrive/Mathematics/TFG/Latex/Figs/window.png', transparent=True)
plt.show()
