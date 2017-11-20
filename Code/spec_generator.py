#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:43:34 2017

@author: jacob
"""
 
import numpy as np
import scipy.signal
import os
import librosa
from sklearn.preprocessing import normalize


path_to_8k = '/media/jacob/Files/UrbanSound8K'
save_path = path_to_8k+'/spectograms'
path = path_to_8k+'/audio'

os.mkdir(save_path)    

frac = 1./4 #Fraction of energy that must be in a segment, compared to if energy was uniformly distributed
i = 0
j = 0
folders = os.listdir(path)
for folder in folders:
    if folder[0:4] == 'fold':
        os.mkdir(save_path+'/'+folder)
        path_temp = path+'/'+folder
        files = os.listdir(path_temp)
        for file in files:
            if file[-4:] == '.wav':
                try:
                    sound, sample_rate = librosa.load(path_temp+'/'+file)
                    f, t, sxx = scipy.signal.spectrogram(sound, sample_rate, noverlap=64)
                    sxx = normalize(sxx[1:])
                    specs = int((np.shape(sxx)[1]-np.shape(sxx)[1]%128)/128)
                    energy = sum(sum(sxx))
                    threshold = float(energy)/specs * frac
                    for i in range(specs):
                        temp = sxx[:,(i)*128:(i+1)*128]
                        if sum(sum(temp)) > threshold:
                            np.savetxt(save_path+'/'+folder+'/'+file[:-4]+'_'+str(i)+'.txt', temp, delimiter=',')
                        else:
                            j += 1
                except:
                    i += 1
  
                    