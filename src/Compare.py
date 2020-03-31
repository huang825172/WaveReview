import scipy.io.wavfile
from scipy.fftpack import fft, ifft
from scipy import signal
import numpy as np
import pylab as pl
import math

files = [
    ['waves/Des/2.wav','b'],
    ['waves/Ori/2.wav','g']
]

for f in files:
    rate, data = scipy.io.wavfile.read(f[0])
    # Divide into frames
    frames = []
    fps = 40
    frame_point = int(rate / fps)
    if type(data[0])!=np.int16:
        data = [d[0] for d in data]
    while True:
        if len(data) > frame_point:
            frames.append(data[:frame_point])
            data = data[frame_point + 1:]
        else:
            newFrame = []
            for i in range(len(data)):
                newFrame.append(data[i])
            for _ in range(frame_point - len(data)):
                newFrame.append(0.0)
            frames.append(newFrame)
            break
    # FFT
    for index in range(len(frames)):
        frame = []
        # Hanning
        for n in range(len(frames[index])):
            frame.append(
                frames[index][n] * \
                0.5 * (1 - math.cos(2 * math.pi * (n - 1) / len(frames[index]))))
        fft_res = fft(frame)
        amp = np.abs(fft_res)
        pha = np.angle(fft_res)
        b, a = signal.butter(8, 0.2)
        Mel = [2595 * math.log(1 + f / 700, 10) for f in amp[:int(len(amp) / 2)]]
        lHk = signal.filtfilt(b, a, np.log(Mel))
        lEk = np.log(Mel) - lHk
        pl.subplot(211)
        pl.plot(np.power(10, lEk) - 1, linewidth=0.1, c=f[1])
        pl.subplot(212)
        # pl.plot(np.power(10, lHk) - 1, linewidth=0.1, c=f[1])
        pl.plot(amp, linewidth=0.1, c=f[1])
pl.show()
