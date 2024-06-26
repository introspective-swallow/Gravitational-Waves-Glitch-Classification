import matplotlib.pyplot as plt
import numpy as np
import math

L = 25
N=512
n = np.arange(L)
w0=math.pi / 2
x = np.cos(w0 * n)

xfft1 =np.fft.fft(x, N)

freq1 = 2*math.pi* np.fft.fftfreq(N)

# Hann window
w = np.sin(math.pi * n / L)**2
x2 = x * w

xfft2 =np.fft.fft(x2, N)

freq2 = 2*math.pi* np.fft.fftfreq(N)

fig, axs = plt.subplots(nrows=1, ncols=2)

axs[0].plot(np.fft.fftshift(freq1), np.sqrt(xfft1**2))
axs[0].set_yticks([])
axs[0].set_xticks([-math.pi,-math.pi/2,0,math.pi/2,math.pi], ['-π', '-π/2', '0', 'π/2', 'π'])

axs[1].plot(np.fft.fftshift(freq2), np.sqrt(xfft2**2))
axs[1].set_xticks([-math.pi,-math.pi/2,0,math.pi/2,math.pi], ['-π', '-π/2', '0', 'π/2', 'π'])
axs[1].set_yticks([])

plt.show()
