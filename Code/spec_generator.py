#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:43:34 2017

@author: jacob
"""
 
import numpy as np
import scipy.signal
import os
import utils
import librosa
from sklearn.preprocessing import normalize
import random


path_to_8k = '/media/jacob/Files/UrbanSound8K'
save_path = path_to_8k+'/spectrograms'
save_path_log= path_to_8k+'/log_spectrograms'
save_path_noise = path_to_8k+'/noise_spectrograms'
save_path_log_noise = path_to_8k+'/log_noise_spectrograms'
path = path_to_8k+'/audio'


os.mkdir(save_path)    
os.mkdir(save_path_log)
os.mkdir(save_path_noise)
os.mkdir(save_path_log_noise)

white_noise_db = 6
    
shuffle_folds = True
test_frac = 0.1

frac = 1./3 #Fraction of energy that must be in a segment, compared to if energy was uniformly distributed
i = 0
j = 0
folders = os.listdir(path)
for folder in folders:
    if folder[0:4] == 'fold':
        os.mkdir(save_path+'/'+folder)
        os.mkdir(save_path_log+'/'+folder)
        os.mkdir(save_path_noise+'/'+folder)
        os.mkdir(save_path_log_noise+'/'+folder)
        path_temp = path+'/'+folder
        files = os.listdir(path_temp)
        for file in files:
            if file[-4:] == '.wav':
                try:
                    sound, sample_rate = librosa.load(path_temp+'/'+file)
                    var_sound = np.var(sound)
                    sound_w_noise = np.zeros(len(sound))
                    f, t, sxx = scipy.signal.spectrogram(sound, sample_rate, noverlap=64)
                    log_sxx = normalize(librosa.power_to_db(sxx[1:], ref=np.max))
                    sxx = normalize(sxx[1:])
                    specs = int((np.shape(sxx)[1]-np.shape(sxx)[1]%128)/128)
                    energy = sum(sum(sxx))
                    threshold = float(energy)/specs * frac
                    var_noise = var_sound*(10**(white_noise_db/10.))**(-1)
                    noise = np.random.normal(0,var_noise,len(sound))
                    sound_w_noise = sound + noise
                    f_noise, t_noise, sxx_noise = scipy.signal.spectrogram(sound_w_noise)
                    sxx_noise = normalize(sxx_noise[1:])
                    log_sxx_noise = normalize(librosa.power_to_db(sxx[1:], ref=np.max))
                    for i in range(specs):
                        temp = sxx[:,(i)*128:(i+1)*128]
                        temp_log = log_sxx[:,(i)*128:(i+1)*128]
                        temp_noise = sxx_noise[:,(i)*128:(i+1)*128]
                        temp_log_noise = log_sxx_noise[:,(i)*128:(i+1)*128]
                        if sum(sum(temp)) > threshold:
                            np.savetxt(save_path+'/'+folder+'/'+file[:-4]+'_'+str(i)+'.txt', temp, delimiter=',')
                            np.savetxt(save_path_log+'/'+folder+'/'+file[:-4]+'_'+str(i)+'.txt', temp_log, delimiter=',')
                            np.savetxt(save_path_noise+'/'+folder+'/'+file[:-4]+'_'+str(i)+'.txt', temp_noise, delimiter=',')
                            np.savetxt(save_path_log_noise+'/'+folder+'/'+file[:-4]+'_'+str(i)+'.txt', temp_log_noise, delimiter=',')
                        else:
                            j += 1
                except:
                    i += 1
  
if shuffle_folds == True:
    for i in range(10):
        os.mkdir(save_path+'/fold'+str(i+11))
        os.mkdir(save_path_log+'/fold'+str(i+11))
        os.mkdir(save_path_log_noise+'/fold'+str(i+11))
        os.mkdir(save_path_noise+'/fold'+str(i+11))
    for i in range(10):
        files = os.listdir(save_path+'/fold'+str(i+1))
        files_log = os.listdir(save_path_log+'/fold'+str(i+1))
        files_noise = os.listdir(save_path_noise+'/fold'+str(i+1))
        files_log_noise = os.listdir(save_path_log_noise+'/fold'+str(i+1))
        random.shuffle(files)
        random.shuffle(files_log)
        random.shuffle(files_noise)
        random.shuffle(files_log_noise)
        length = int(len(files)*test_frac)
        for k in range(10):
            for j in range(length):
                os.rename(save_path+'/fold'+str(i+1)+'/'+files[k*length+j],save_path+'/fold'+str(k+11)+'/'+files[k*length+j])
                os.rename(save_path_log+'/fold'+str(i+1)+'/'+files_log[k*length+j],save_path_log+'/fold'+str(k+11)+'/'+files_log[k*length+j])
                os.rename(save_path_noise+'/fold'+str(i+1)+'/'+files_noise[k*length+j],save_path_noise+'/fold'+str(k+11)+'/'+files_noise[k*length+j])
                os.rename(save_path_log_noise+'/fold'+str(i+1)+'/'+files_log_noise[k*length+j],save_path_log_noise+'/fold'+str(k+11)+'/'+files_log_noise[k*length+j])