import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

x = np.linspace(0, 1, 1400)
y = 7*np.sin(2*np.pi*200*x) + 5*np.sin(2*np.pi*400*x)+3*np.sin(2*np.pi*600*x)

# plt.figure()
# # plt.plot(x, y)
# plt.plot(x[0:50], y[0:50])
# plt.title('Original Wave (0-50)')
# plt.show()

N=1400
x = np.arange(N)

fft_y = fft(y)
abs_y = np.abs(fft_y)
angle_y = np.angle(fft_y)

plt.figure()
plt.plot(x, abs_y)
plt.title('ZF')

plt.figure()
plt.plot(x, angle_y)
plt.title('XW')

plt.show()