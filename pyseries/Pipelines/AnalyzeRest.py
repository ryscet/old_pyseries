# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:32:36 2016

@author: user
"""
import sys
sys.path.insert(0, '/Users/user/Desktop/repo_for_pyseries/pyseries/pyseries')

import LoadingData as loading
import obspy.signal.filter as filters

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import signal
import matplotlib.pyplot as plt



channels = loading.Read_edf.Combine_EDF_XML('/Users/user/Desktop/Nagrania/rest/Agnieszka_03_06/')

slices = Make_Slices_for_Channel(channels, ['EEG O1'],0, 500* 140)


event = PlotPowerSpectrum(slices['EEG O1'])

def mark_events(channels):
    
    fig, axes = plt.subplots(1,1)
    
    sig  = filters.bandpass(channels["EEG O1"], 2, 30, df = 500)
    axes.plot(channels["timestamp"], sig)
    
    for idx, row in channels["events"].iterrows():
        axes.axvline(idx, color='r', linestyle='--')
        #print(row.index)
  

mark_events(channels)

      
def PlotPowerSpectrum(electrode_slices):
    sns.set()
    fig, axes = plt.subplots(1)
    for name, event in electrode_slices.items():
        f, Pxx_den = signal.welch(event, 500, nperseg=32)
        #avg_Pxx = np.mean(Pxx_den, axis = 0)        
        
        if('condtions_Pxx' not in locals()):
            condtions_Pxx = Pxx_den
        else:
            condtions_Pxx = np.dstack((condtions_Pxx, Pxx_den))       
    
    sns.tsplot(data=condtions_Pxx, time = f,  err_style="unit_traces", condition = [key for key in electrode_slices.keys()], ax = axes)
                     
    axes.set_yticklabels(labels = f, rotation = 0)
    
    axes.set_ylabel('Welch Power Density')
    axes.set_xlabel('frequency')
    
    return condtions_Pxx


def Make_Slices_for_Channel(channels, ch_names,n_samples_back, n_samples_forth):
    events = channels["events"]
    
    electrode_slices = {}
    
    for name in ch_names:
        filtered = filters.bandpass(channels[name], 2, 30, df = 500)
        
        
        time_series =pd.Series(np.array(filtered, dtype = 'float32'), index = channels['timestamp'] )
        electrode_slices[name] = Make_Slices_Groups(time_series, events, n_samples_back, n_samples_forth)
    
    return electrode_slices
    
def Make_Slices_Groups(data, events, n_samples_back, n_samples_forth):
    """Loads signal and events and creates a list of np.arrays per event type
    
        Parameters
        ----------
        data: np.array
            whole EEG signal
        events: DataFrame
            Timestamps and description of events
        n_samples_back, n_samples_forth: (int, int)
            window size around time of event
        
        Returns
        -------
        grouped slices: list(np.arra)
            list of arrays, one per event type
        
    """

    grouped_slices = {}   
    
    for name, events_single_type in events.groupby('code'):
        grouped_slices[str(name)] = Cut_Slices_From_Signal(data, events_single_type, n_samples_back, n_samples_forth)    
    return grouped_slices
    
def Cut_Slices_From_Signal(signal, events, n_samples_back, n_samples_forth):
#Iterates through all events and finds the index of the closest EEG sample in the signal. 
#Takes a window spanning back and forth around the closest sample. 
    slices = np.zeros((len(events.index), n_samples_back + n_samples_forth))    
    for idx, timestamp in enumerate(events.index):
        #Finding first index greater than event timestamp produces more error than finding the closest data timestamp regrdless of whether it is larger or smaller than event.
        event_index = Find_Closest_Sample(signal,timestamp)
        print('event index: %i'%event_index)
        #TODO check how inclusion/exclusion of lower and iupper bounds might affect the seslected slice size
        slices[idx,:] = signal.iloc[event_index - n_samples_back : event_index + n_samples_forth]
    
    return slices
    
def Find_Closest_Sample(df, dtObj):
    return np.argmin(np.abs(df.index - dtObj))