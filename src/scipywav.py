import scipy.io.wavfile
from scipy.fftpack import fft, ifft
import numpy as np
import pylab as pl
import math

rate, data = scipy.io.wavfile.read('waves/IMKiana.wav')
rate1, data1 = scipy.io.wavfile.read('waves/IMMei.wav')
r = fft(data)
amp = np.abs(r)
pha = np.angle(r)
r1 = fft(data1)
amp1 = np.abs(r1)
pha1 = np.angle(r1)
pl.subplot(311)
pl.plot(amp, linewidth=0.1)
pl.subplot(312)
pl.plot(amp1, linewidth=0.1)
for i in range(len(amp)):
        #amp[int((i/len(amp1))*(len(amp)-1))] = amp1[i]
        amp[i] = amp[i]+amp1[int((i/len(amp))*(len(amp1)-1))]*2
        amp[i] /= 3
pl.subplot(313)
pl.plot(amp, linewidth=0.1)
pl.show()
rifft = []
for i in range(len(r)):
    rifft.append(complex(
        amp[i]*math.cos(pha[i]),
        amp[i]*math.sin(pha[i])))
r = ifft(rifft).astype(float)/32864
scipy.io.wavfile.write('waves/out.wav', rate, r)
