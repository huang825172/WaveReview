# Re-implement of the paper from Columbia University 2003
# Technological process:
# 1.DTW
# 2.LPC
# 3.Vocal frame classification
# 4.Pitch estimation
# 5.Pitch shift
# 6.inverse LPC
# Train: 1-4
# Synthetic: 5-6
# Training data pair: Ori/2.wav Des/2.wav
# Test data: Ori/3.wav

import numpy as np
import pylab as pl
import librosa as lr
import scipy.io.wavfile
import scipy.signal
from scipy.fftpack import fft, ifft
import time
import json
import math

params = {
    '3': {
        'ori_file':'Ori/3.wav',
        'des_file':'Des/3.wav',
        'ori_segs_mark':[[2,3500,0],[2,0,0],[1,0,0]],
        'des_segs_mark':[[1,0,0],[0,0,0],[2,0,0],[1,0,0]]
    }
}
ori_file = params['3']['ori_file']
des_file = params['3']['des_file']
ori_segs_mark = params['3']['ori_segs_mark']
des_segs_mark = params['3']['des_segs_mark']

def devideFrames(data, rate, fps):
    frames = []
    frame_point = int(rate / fps)
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
    return np.array(frames)

def segsVoiced(data):
    e = 0
    for d in data:
        e += math.fabs(d)
    e /= len(data)
    voi = []
    for d in data:
        voi.append(0 if math.fabs(d)<e*0.1 else 1)
    for i in range(110, len(voi)-110):
        if voi[i]==0:
            if np.count_nonzero(voi[i-100:i+100]) > 110:
                voi[i]=1
    voip = 0
    voi_segs = []
    tot = 0
    while voip<len(voi)-1:
        while voi[voip] == 0 and voip<len(voi)-1:
            voip += 1
        c = 0
        b = voip
        while voi[voip] == 1:
            c += 1
            voip += 1
        voi_segs.append((b, c))
        tot += c
    tot /= len(voi_segs)
    for seg in voi_segs:
        if seg[1]<tot*1.3:
            for voip in range(seg[0], seg[0]+seg[1]):
                voi[voip] = 0
    voip = 0
    voi_segs = []
    while voip < len(voi) - 1:
        while voi[voip] == 0 and voip < len(voi) - 1:
            voip += 1
        c = 0
        b = voip
        while voi[voip] == 1:
            c += 1
            voip += 1
        voi_segs.append((b, c))
    return np.array(voi_segs)

def alignSegments(ori_data, des_data, ori_mark, des_mark):
    ori_segs = segsVoiced(ori_data)
    des_segs = segsVoiced(des_data)
    ori_segs_data = []
    des_segs_data = []
    segp = 0
    for mark in ori_mark:
        data = []
        if mark[0]==0:
            segp += 1
            continue
        for _ in range(mark[0]):
            for i in range(
                    ori_segs[segp][0] + mark[1],
                    ori_segs[segp][0] + ori_segs[segp][1] + mark[2]):
                data.append(ori_data[i])
            segp += 1
        ori_segs_data.append(data)
    segp = 0
    for mark in des_mark:
        data = []
        if mark[0]==0:
            segp += 1
            continue
        for _ in range(mark[0]):
            for i in range(
                    des_segs[segp][0] + mark[1],
                    des_segs[segp][0] + des_segs[segp][1] + mark[2]):
                data.append(des_data[i])
            segp += 1
        des_segs_data.append(data)
    if len(ori_segs_data) != len(des_segs_data):
        return None
    return np.array([(ori_segs_data[i], des_segs_data[i]) for i in range(len(ori_segs_data))])

def decompose(frame):
    coefficients = lr.lpc(frame, 12)
    lfilter = scipy.signal.lfilter(
        [0] + -1 * coefficients[1:], [1],
        frame)
    excitation = frame - lfilter
    Crsd = np.real(ifft(np.abs(fft(lfilter))))
    maxI = 0
    maxV = 0
    for i_crsd in range(25, len(Crsd)):
        if Crsd[i_crsd] > maxV:
            maxV = Crsd[i_crsd]
            maxI = i_crsd
    maxVK = 0
    for i_crsd in range(25, maxI - 25):
        if Crsd[i_crsd] > maxVK:
            maxVK = Crsd[i_crsd]
    pitch = maxVK if maxVK > maxV * 0.7 else maxV
    return coefficients, pitch

if __name__ == '__main__':
    ori_rate, ori_data = scipy.io.wavfile.read(ori_file)
    ori_data = np.array([d[0] for d in ori_data])
    des_rate, des_data = scipy.io.wavfile.read(des_file)
    pairs = alignSegments(
        ori_data, des_data,
        ori_segs_mark, des_segs_mark)
    result = []
    for i in range(len(pairs)):
        ori_frames = devideFrames(pairs[i][0], ori_rate, 50)
        des_frames = devideFrames(pairs[i][1], des_rate, 50)
        if len(ori_frames)>len(des_frames):
            for i_ori in range(len(ori_frames)):
                i_des = int(len(des_frames)*(i_ori/len(ori_frames)))
                oc, op = decompose(ori_frames[i_ori])
                dc, dp = decompose(des_frames[i_des])
                result.append([list(oc), op, list(dc), dp])
        else:
            for i_des in range(len(des_frames)):
                i_ori = int(len(ori_frames)*(i_des/len(des_frames)))
                oc, op = decompose(ori_frames[i_ori])
                dc, dp = decompose(des_frames[i_des])
                result.append([list(oc), op, list(dc), dp])
    lpc_funcs = np.zeros(shape=(len(result[0][2]), len(result), 2))
    for i_r in range(len(result)):
        for i_dc in range(len(result[i_r][2])):
            lpc_funcs[i_dc][i_r] = [result[i_r][0][i_dc], result[i_r][2][i_dc]]
    lpc_trans = []
    for i_dc in range(len(result[i_r][2])):
        x = [ lpc_funcs[i_dc][i_r][1] for i_r in range(len(result))]
        y = [ lpc_funcs[i_dc][i_r][0] for i_r in range(len(result))]
        A = np.polyfit(x, y, 2)
        tran = np.poly1d(A)
        pl.plot(tran(np.arange(1400,300000)))
        lpc_trans.append(tran)
    pl.show()
    p_funcs = np.zeros(shape=(len(result),2))
    for i_r in range(len(result)):
        p_funcs[i_r] = [result[i_r][1], result[i_r][3]]
    x = [ p_funcs[i_r][0] for i_r in range(len(result))]
    y = [ p_funcs[i_r][1] for i_r in range(len(result))]
    A = np.polyfit(x, y, 2)
    tran = np.poly1d(A)
    pl.plot(tran(np.arange(10000, 20000)))
    pl.show()
    # ori_frames = devideFrames(pairs[1][0], ori_rate, 50)
    # new_data = np.zeros(len(pairs[1][0])+1000, dtype=type(pairs[1][0][0]))
    # index = 0
    # pitchs = []
    # for i_ori in range(len(ori_frames)):
    #     frame = ori_frames[i_ori]
    #     c, p = decompose(frame)
    #     lfilter = scipy.signal.lfilter(
    #         [0] + -1 * c[1:], [1],
    #         frame)
    #     frame -= lfilter
    #     pitchs.append(p)
    #     for d in frame:
    #         new_data[index] = d
    #         index += 1
    # pl.subplot(211)
    # pl.plot(pitchs, linewidth=1)
    # pl.subplot(212)
    # pl.plot(new_data, linewidth=1)
    # pl.show()
    # scipy.io.wavfile.write('out.wav', ori_rate, new_data)
    # des_funcs = np.zeros(shape=(len(result[0][2]), len(result), 2))
    # ori_funcs = np.zeros(shape=(len(result[0][2]), len(result), 2))
    # des_ppower = 0.0
    # for i_r in range(len(result)):
    #     for i_dc in range(len(result[i_r][2])):
    #         des_funcs[i_dc][i_r] = [result[i_r][2][i_dc], result[i_r][3]]
    #         ori_funcs[i_dc][i_r] = [result[i_r][0][i_dc], result[i_r][1]]
    #     des_ppower += result[i_r][3]/result[i_r][1]
    # des_ppower /= len(result)
    # des_lpc_trans = []
    # for i_dc in range(len(result[0][2])):
    #     x = [ des_funcs[i_dc][i_r][1] for i_r in range(len(result))]
    #     y = [ des_funcs[i_dc][i_r][0] for i_r in range(len(result))]
    #     A = np.polyfit(x, y, 2)
    #     tran = np.poly1d(A)
    #     pl.plot(tran(np.arange(1429,120000)),c='b',linewidth=1)
    #     des_lpc_trans.append(tran)
    # ori_lpc_trans = []
    # for i_dc in range(len(result[0][2])):
    #     x = [ ori_funcs[i_dc][i_r][1] for i_r in range(len(result))]
    #     y = [ ori_funcs[i_dc][i_r][0] for i_r in range(len(result))]
    #     A = np.polyfit(x, y, 2)
    #     tran = np.poly1d(A)
    #     pl.plot(tran(np.arange(1429,120000)),c='r',linewidth=1)
    #     ori_lpc_trans.append(tran)
    # pl.show()
    # ori_frames = devideFrames(pairs[1][0], ori_rate, 50)
    # new_data = []
    # for i_ori in range(len(ori_frames)):
    #     frame = ori_frames[i_ori]
    #     coefficients = lr.lpc(frame, 12)
    #     lfilter = scipy.signal.lfilter(
    #         [0] + -1 * coefficients[1:], [1],
    #         frame)
    #     excitation = frame - lfilter
    #     _, p = decompose(frame)
    #     newcoefficients = [des_lpc_trans[i_dc](p*des_ppower) for i_dc in range(len(result[0][2]))]
    #     # newcoefficients = [ori_lpc_trans[i_dc](p) for i_dc in range(len(result[0][2]))]
    #     newlfilter = scipy.signal.lfilter(
    #         [0] + -1 * newcoefficients[1:], [1],
    #         frame)
    #     signal = excitation + newlfilter
    #     for s in signal:
    #         new_data.append(s)
    # new_data = np.array(new_data)
    # new_data = lr.effects.pitch_shift(new_data, int(ori_rate/2), 0.1)
    # # new_data = np.array(new_data)*5/2048
    # # b, a = scipy.signal.butter(8, 0.8, 'lowpass')
    # # new_data = scipy.signal.filtfilt(b, a, new_data)
    # pl.subplot(211)
    # pl.plot(pairs[1][0], linewidth=1)
    # pl.subplot(212)
    # pl.plot(new_data, linewidth=1)
    # pl.show()
    # scipy.io.wavfile.write('out.wav', int(ori_rate/2), new_data)
