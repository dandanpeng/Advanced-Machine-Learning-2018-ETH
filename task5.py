#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:07:14 2018

@author: pengdandan
"""

import pandas as pd
import numpy as np
from biosppy.signals import eeg

train_eeg1 = pd.read_csv('train_eeg1.csv')
train_eeg2 = pd.read_csv('train_eeg2.csv')
train_emg = pd.read_csv('train_emg.csv')
label = pd.read_csv('train_labels.csv')
test_eeg1 = pd.read_csv('test_eeg1.csv')

train_eeg = pd.concat([train_eeg1.iloc[:,1:],train_eeg2.iloc[:,1:],train_emg.iloc[:,1:]],axis = 1)

def get_result(array):
    a1 = array[0:int(len(array)/2)]
    a2 = array[int(len(array)/2):]
    matrix = np.vstack((a1,a2)).T
    out = eeg.eeg(signal = matrix,sampling_rate = 128,show = False)
    return np.sqrt(np.sum(np.power(out['theta'][:,0],2))/31),np.sqrt(np.sum(np.power(out['alpha_low'][:,0],2))/31),
np.sqrt(np.sum(np.power(out['alpha_high'][:,0],2))/31),np.sqrt(np.sum(np.power(out['beta'][:,0],2))/31),
np.sqrt(np.sum(np.power(out['gamma'][:,0],2))/31)


theta = train_eeg.iloc[:,1:].apply(get_result,axis = 1)

eeg1_a = train_eeg1.iloc[range(0,64800,5),1:]
eeg1_b = train_eeg1.iloc[range(1,64800,5),1:]
eeg1_c = train_eeg1.iloc[range(2,64800,5),1:]
eeg1_d = train_eeg1.iloc[range(3,64800,5),1:]
eeg1_e = train_eeg1.iloc[range(4,64800,5),1:]

eeg1_merge = np.hstack((eeg1_a,eeg1_b,eeg1_c,eeg1_d,eeg1_e))

eeg2_a = train_eeg2.iloc[range(0,64800,5),1:]
eeg2_b = train_eeg2.iloc[range(1,64800,5),1:]
eeg2_c = train_eeg2.iloc[range(2,64800,5),1:]
eeg2_d = train_eeg2.iloc[range(3,64800,5),1:]
eeg2_e = train_eeg2.iloc[range(4,64800,5),1:]

eeg2_merge = np.hstack((eeg2_a,eeg2_b,eeg2_c,eeg2_d,eeg2_e))

emg_a = train_emg.iloc[range(0,64800,5),1:]
emg_b = train_emg.iloc[range(1,64800,5),1:]
emg_c = train_emg.iloc[range(2,64800,5),1:]
emg_d = train_emg.iloc[range(3,64800,5),1:]
emg_e = train_emg.iloc[range(4,64800,5),1:]

emg_merge = np.hstack((emg_a,emg_b,emg_c,emg_d,emg_e))

train = np.vstack((eeg1_merge[0,:],eeg2_merge[0,:],emg_merge[0,:]))
for i in range(1,):
    data = np.vstack((eeg1_merge[i,:],eeg2_merge[i,:],emg_merge[i,:]))
    train = np.concatenate((train,data),axis = 0)
    print(i)


prediction = pd.read_csv('prediction1.csv')
id = pd.DataFrame({'Id':test_eeg1.iloc[:,0]})
id['y'] = prediction
id.iloc[43199,1] = 1
id.to_csv('prediction1.csv',index = False)


from scipy import signal
f,Pxx_den = signal.welch(eeg1_merge[0],fs = 128,scaling = 'spectrum')
plt.semilogy(f, Pxx_den)
eeg.eeg(signal = np.array(signal),sampling_rate = 128)
signal = pd.DataFrame(train_eeg1.iloc[0,1:])
signal[1] = train_eeg2.iloc[0,1:]
a = np.array(signal)

plt.semilogy(f[16:32], Pxx_den[16:32])

def calc_bands_power(x):
    f, psd = signal.welch(x, fs = 128)
    #power = {band: np.mean(psd[np.where((f >= lf) & (f <= hf))]) for band, (lf, hf) in bands.items()}
    delta_power = np.mean(psd[np.where((f >= 0.8) & (f < 5))])
    theta_power = np.mean(psd[np.where((f >= 5) & (f < 8.6))])
    alpha_power = np.mean(psd[np.where((f >= 8.6) & (f < 12))])
    spindles = np.mean(psd[np.where((f >= 11) & (f <= 15))])
    beta_power = np.mean(psd[np.where((f >= 16) & (f < 30))])
    gamma_power = np.mean(psd[np.where((f >= 30) & (f <= 40))])
    return (delta_power,theta_power,alpha_power,spindles,beta_power,gamma_power)


from scipy.fftpack import rfft, irfft, fftfreq 
W = fftfreq(2056, d= 1/128) 
f_signal = rfft(eeg1_merge[0]) 

cut_f_signal = f_signal.copy() 
cut_f_signal[(W<2)] = 0 

cut_signal = irfft(cut_f_signal) 

eeg1.to_csv('eeg1_signal.csv',index = False)

