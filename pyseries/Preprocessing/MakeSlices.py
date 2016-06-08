# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:20:45 2016

@author: user
"""
import obspy.signal.filter as filters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mark_events(channels, ch_names):
    
    for name in ch_names:
        fig, axes = plt.subplots(1,1)
        
        sig  = filters.bandpass(channels["EEG O1"], 2, 30, df = 500)
        axes.plot(channels["timestamp"], sig)
        
        for idx, row in channels["events"].iterrows():
            axes.axvline(idx, color='r', linestyle='--')
            #print(row.index)
            
    
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
    
def Cut_Slices_From_Signal(_signal, events, n_samples_back, n_samples_forth):
#Iterates through all events and finds the index of the closest EEG sample in the _signal. 
#Takes a window spanning back and forth around the closest sample. 
    slices = np.zeros((len(events.index), n_samples_back + n_samples_forth))    
    for idx, timestamp in enumerate(events.index):
        #Finding first index greater than event timestamp produces more error than finding the closest data timestamp regrdless of whether it is larger or smaller than event.
        event_index = Find_Closest_Sample(_signal,timestamp)
        print('event index: %i'%event_index)
        #TODO check how inclusion/exclusion of lower and iupper bounds might affect the seslected slice size
        slices[idx,:] = _signal.iloc[event_index - n_samples_back : event_index + n_samples_forth]
    
    return slices
    
def Find_Closest_Sample(df, dtObj):
    return np.argmin(np.abs(df.index - dtObj))