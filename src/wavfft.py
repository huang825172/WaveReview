import wave
import pylab as pl
import numpy as np
from scipy.fftpack import fft

f = wave.open(r'waves/IMKiana.wav', 'rb')
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]

str_data = f.readframes(nframes)
f.close()

wave_data = np.fromstring(str_data, dtype=np.short)
wave_data.shape = -1, 2
wave_data = wave_data.T
time = np.arange(0, nframes / 2) * (1.0 / framerate)

fft_l = fft(wave_data[0])
abs_l = np.abs(fft_l)
angle_l = np.angle(fft_l)

pl.subplot(2,1,1)
pl.plot(time, wave_data[0], linewidth=0.1)
pl.xlabel('time(sec)')
pl.ylabel('Pitch')
pl.title('Original')
pl.subplot(2,1,2)
pl.scatter(abs_l, angle_l, s=0.1)
pl.xlabel('Amplitude')
pl.ylabel('Phase')
pl.title('Image')

pl.show()
