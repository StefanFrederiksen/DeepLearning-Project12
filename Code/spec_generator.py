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
path = path_to_8k+'/audio'

os.mkdir(save_path)    
os.mkdir(save_path_log)

    
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
        path_temp = path+'/'+folder
        files = os.listdir(path_temp)
        for file in files:
            if file[-4:] == '.wav':
                try:
                    sound, sample_rate = librosa.load(path_temp+'/'+file)
                    f, t, sxx = scipy.signal.spectrogram(sound, sample_rate, noverlap=64)
                    #sxx = librosa.feature.melspectrogram(sound, sr=sample_rate,n_fft=1024)
                    log_sxx = normalize(librosa.power_to_db(sxx[1:], ref=np.max))
                    sxx = normalize(sxx[1:])
                    specs = int((np.shape(sxx)[1]-np.shape(sxx)[1]%128)/128)
                    energy = sum(sum(sxx))
                    threshold = float(energy)/specs * frac
                    for i in range(specs):
                        temp = sxx[:,(i)*128:(i+1)*128]
                        temp_log = log_sxx[:,(i)*128:(i+1)*128]
                        if sum(sum(temp)) > threshold:
                            np.savetxt(save_path+'/'+folder+'/'+file[:-4]+'_'+str(i)+'.txt', temp, delimiter=',')
                            np.savetxt(save_path_log+'/'+folder+'/'+file[:-4]+'_'+str(i)+'.txt', temp_log, delimiter=',')
                        else:
                            j += 1
                except:
                    i += 1
  
if shuffle_folds == True:
    os.mkdir(save_path+'/fold11')
    os.mkdir(save_path_log+'/fold11')
    for i in range(10):
        files = os.listdir(save_path+'/fold'+str(i+1))
        files_log = os.listdir(save_path_log+'/fold'+str(i+1))
        random.shuffle(files)
        random.shuffle(files_log)
        length = int(len(files)*test_frac)
        for j in range(length):
            os.rename(save_path+'/fold'+str(i+1)+'/'+files[j],save_path+'/fold11/'+files[j])
            os.rename(save_path_log+'/fold'+str(i+1)+'/'+files_log[j],save_path_log+'/fold11/'+files_log[j])
    