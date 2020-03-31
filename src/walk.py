import os
import wave
import pylab as pl
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import stats
import math

names = ['WaveReview', 'Mei', 'Bronya', 'Himeko', 'Theresa', 'Sakura']
names = ['Sakura']


def normalize(data):
    m = np.mean(data)
    mx = max(data)
    mn = min(data)
    return [(float(i) - m) / (mx - mn) for i in data]


Gamp = []
Gpha = []
Gdata = []
Gfft = []
Gfr = 0.0

if __name__ == '__main__':
    rootDir = '/home/huang825172/BH3AudioPack/NAME/NAME_Bridge_Cover'
    for name in names:
        realDir = rootDir.replace('NAME', name)
        list = os.listdir(realDir)
        result_amp = np.full((35000,), 0.5)
        for i in range(len(list)):
            item = list[i]
            path = os.path.join(realDir, item)
            if os.path.isfile(path):
                if i != 10:
                    continue
                f = wave.open(path, 'rb')
                params = f.getparams()
                _, _, framerate, nframes = params[:4]
                str_data = f.readframes(nframes)
                f.close()
                wave_data = np.fromstring(str_data, dtype=np.short)
                wave_data.shape = -1, 2
                data = wave_data.T
                # start = int(len(data[0])/4)
                # end = start*2
                start = 0
                end = len(data[0]) - 1
                data_fft = fft((data[0][start:end] + data[1][start:end]) / 2)
                amp = normalize(np.abs(data_fft))
                pha = normalize(np.angle(data_fft))
                amp_cdf = stats.norm.cdf(amp[:int(len(amp) / 2)])
                if name == 'Sakura' and i == 10:
                    print(item)
                    Gpha = np.angle(data_fft)
                    Gamp = np.abs(data_fft)
                    Gdata = data
                    Gfft = data_fft
                    Gfr = framerate
                # for ia in range(len(amp_cdf)):
                #     if i!=0:
                #         result_amp[ia] += amp_cdf[ia]
                #         result_amp[ia] /= 2
                #     else:
                #         result_amp[ia] = amp_cdf[ia]
                # x = np.arange(len(amp_cdf))
                # A = np.polyfit(x, amp_cdf,15)
                # fx = np.poly1d(A)
                # Y = fx(x)
                # pl.subplot(211)
                # pl.scatter(amp, pha, s=0.1)
                # pl.title(name)
                # pl.subplot(212)
                # pl.plot(x, Y, linewidth=1)
                # pl.show()
        # x = np.arange(len(result_amp))
        # A = np.polyfit(x, result_amp,15)
        # fx = np.poly1d(A)
        # Y = fx(x)
        # pl.plot(result_amp,linewidth=0.1)
        # pl.title(name)
        # pl.show()
    Gifft = []
    for index in range(len(Gpha)):
        Gifft.append(complex(Gamp[index] * math.cos(Gpha[index]), Gamp[index] * math.sin(Gpha[index])))
    Gres = ifft(Gifft).astype(float)/32864
    pl.subplot(211)
    pl.plot(data[0], linewidth=0.1)
    pl.subplot(212)
    pl.plot(Gres, linewidth=0.1)
    pl.show()
    ff = wave.open(r'../out.wav', 'wb')
    ff.setnchannels(1)
    ff.setsampwidth(2)
    ff.setframerate(Gfr / 2)
    ff.writeframes(Gres.tostring())
    ff.close()
