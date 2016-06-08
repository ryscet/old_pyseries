# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:48:03 2016

@author: user
"""

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np



def PlotPowerSpectrum(electrode_slices):
    sns.set()
    sns.set_palette("hls")
    fig, axes = plt.subplots(1)
    for name, event in electrode_slices.items():
        f, Pxx_den = signal.welch(event, 500, nperseg=256)
        #avg_Pxx = np.mean(Pxx_den, axis = 0)        
        
        if('condtions_Pxx' not in locals()):
            condtions_Pxx = Pxx_den
        else:
            condtions_Pxx = np.dstack((condtions_Pxx, Pxx_den))       
    
    sns.tsplot(data=condtions_Pxx[:, 0:20, :], time = f[0:20],  err_style="unit_traces", condition = [key for key in electrode_slices.keys()], ax = axes)
                     
    axes.set_yticklabels(labels = f, rotation = 0)
    
    axes.set_ylabel('Welch Power Density')
    axes.set_xlabel('frequency')
    
    return f, condtions_Pxx
    

    
def PlotErp(electrode_slices, n_back):
    sns.set()
    fig, axes = plt.subplots(1)
    #event_avg = []
    for name, event in electrode_slices.items():
        #event_avg.append(np.mean(event, axis = 0))
        if('events_Erp' not in locals()):
            events_Erp = event
        else:
            events_Erp = np.dstack((events_Erp , event))   
    
    sns.tsplot(data=events_Erp, err_style="unit_traces", condition = [key for key in electrode_slices.keys()], ax = axes)
    axes.axvline(n_back, color='r', linestyle='--')       
    
def PlotSpectrogram(electrode_slices, n_back, n_forth):
    
    sns.set()
    fig, axes = plt.subplots(2,2)
    i = 0
    for name, event in electrode_slices.items():
        #np.flipud, because it's uspide down
        axes[0,i].set_title(name)
        axes[0,0].set_ylabel('> 13 Hz ')
        axes[1,0].set_ylabel('0 - 13 Hz')
        
        #Sxx already after log
        f, t, Sxx, upper_bound = AverageSpectrogram(event)
        
        #Plot the upper part of the spectrum, i.e. above alpha
        g = sns.heatmap(np.flipud(Sxx)[upper_bound::,:], annot=False, robust = True, cbar = True,
                     xticklabels = t, ax = axes[0,i])
        g.set_yticklabels(labels = f[upper_bound::], rotation = 0)        

                     
        #Plot the lower part of the spectrum, i.e. below and including alpha
        g = sns.heatmap(np.flipud(Sxx)[0:upper_bound,:], annot=False, robust = True, cbar = True,
                     xticklabels = t, ax = axes[1,i])
                     
        g.set_yticklabels(labels = f[0:upper_bound], rotation = 0)
                     
        event_pos = CalcEventPosition(n_back, n_forth, len(t))
        
        axes[0,i].axvline(event_pos, color='g', linestyle='--')
        axes[1,i].axvline(event_pos, color='g', linestyle='--')                
        
        i = i +1
        
    return g
    
    
def CalcEventPosition(n_back, n_forth, n_fft_windows):
    pos = n_back / ( (n_back+n_forth) / n_fft_windows )
    #print(pos)
    return pos

def AverageSpectrogram(electrode_event):

    f, t, Sxx = signal.spectrogram(electrode_event, window = 'hamming', fs = 100,nperseg = 64, noverlap = 32, return_onesided =True, scaling = 'spectrum' )
    
    Sxx = np.mean(np.log(Sxx), axis = 0)
    
    upper_bound = np.argmax( f > 13.0 )

    f = [format(x, '.1f') for x in f]
            

    return f, t, Sxx, upper_bound



