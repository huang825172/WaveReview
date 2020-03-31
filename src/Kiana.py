import wave
import pylab as pl
import numpy as np
from scipy.fftpack import fft
from scipy import stats

def normalize(data):
    m = np.mean(data)
    mx = max(data)
    mn = min(data)
    return [(float(i) - m) / (mx - mn) for i in data]

kiana = wave.open(r'waves/IMKiana.wav', 'rb')
params = kiana.getparams()
_, _, framerate, nframes = params[:4]
str_data = kiana.readframes(nframes)
kiana.close()
wave_data = np.fromstring(str_data, dtype=np.short)
wave_data.shape = -1, 2
kiana_data = wave_data.T
kiana_l = fft(kiana_data[0][:framerate])
kiana_amp = normalize(np.abs(kiana_l))
kiana_pha = normalize(np.angle(kiana_l))
kiana_amp_cdf = stats.norm.cdf(kiana_amp[:int(len(kiana_amp)/2)])

kiana2 = wave.open(r'waves/Kiana2.wav', 'rb')
params = kiana2.getparams()
_, _, framerate, nframes = params[:4]
str_data = kiana2.readframes(nframes)
kiana2.close()
wave_data = np.fromstring(str_data, dtype=np.short)
wave_data.shape = -1, 2
kiana2_data = wave_data.T
kiana2_l = fft(kiana2_data[0][:framerate])
kiana2_amp = normalize(np.abs(kiana2_l))
kiana2_pha = normalize(np.angle(kiana2_l))
kiana2_amp_cdf = stats.norm.cdf(kiana2_amp[:int(len(kiana2_amp)/2)])

kiana3 = wave.open(r'waves/Kiana3.wav', 'rb')
params = kiana3.getparams()
_, _, framerate, nframes = params[:4]
str_data = kiana3.readframes(nframes)
kiana3.close()
wave_data = np.fromstring(str_data, dtype=np.short)
wave_data.shape = -1, 2
kiana3_data = wave_data.T
kiana3_l = fft(kiana3_data[0][framerate:framerate*2])
kiana3_amp = normalize(np.abs(kiana3_l))
kiana3_pha = normalize(np.angle(kiana3_l))
kiana3_amp_cdf = stats.norm.cdf(kiana3_amp[:int(len(kiana3_amp)/2)])

mei = wave.open(r'waves/IMMei.wav', 'rb')
params = mei.getparams()
_, _, framerate, nframes = params[:4]
str_data = mei.readframes(nframes)
mei.close()
wave_data = np.fromstring(str_data, dtype=np.short)
wave_data.shape = -1, 2
mei_data = wave_data.T
mei_l = fft(mei_data[0][:framerate])
mei_amp = normalize(np.abs(mei_l))
mei_pha = normalize(np.angle(mei_l))
mei_amp_cdf = stats.norm.cdf(mei_amp[:int(len(mei_amp)/2)])

sakura = wave.open(r'waves/IMSakura.wav', 'rb')
params = sakura.getparams()
_, _, framerate, nframes = params[:4]
str_data = sakura.readframes(nframes)
sakura.close()
wave_data = np.fromstring(str_data, dtype=np.short)
wave_data.shape = -1, 2
sakura_data = wave_data.T
sakura_l = fft(sakura_data[0][:framerate])
sakura_amp = normalize(np.abs(sakura_l))
sakura_pha = normalize(np.angle(sakura_l))
sakura_amp_cdf = stats.norm.cdf(sakura_amp[:int(len(sakura_amp)/2)])

pl.figure(figsize=(12.8,7.2))
pl.subplot(5,1,1)
#pl.scatter(kiana_amp, kiana_pha, s=0.1)
pl.plot(kiana_amp_cdf, linewidth=0.1)
pl.title('WaveReview')
pl.subplot(5,1,2)
#pl.scatter(kiana2_amp, kiana2_pha, s=0.1)
pl.plot(kiana2_amp_cdf, linewidth=0.1)
pl.title('Kiana2')
pl.subplot(5,1,3)
#pl.scatter(kiana3_amp, kiana3_pha, s=0.1)
pl.plot(kiana3_amp_cdf, linewidth=0.1)
pl.title('Kiana3')
pl.subplot(5,1,4)
#pl.scatter(mei_amp, mei_pha, s=0.1)
pl.plot(mei_amp_cdf, linewidth=0.1)
pl.title('Mei')
pl.subplot(5,1,5)
#pl.scatter(sakura_amp, sakura_pha, s=0.1)
pl.plot(sakura_amp_cdf, linewidth=0.1)
pl.title('Sakura')
pl.show()
