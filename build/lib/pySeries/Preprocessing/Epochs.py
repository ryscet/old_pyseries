# -*- coding: utf-8 -*-
"""
MakeSlices
==========

Making epochs (i.e slicing the signal) around events. 
After epoching analysis can be performed, like erp's or spectrograms.
"""
import obspy as ob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mark_events(channels, ch_names):
    """Plots raw signal with event markers on top.
    """

    
    for name in ch_names:
        fig, axes = plt.subplots(1,1)
        
        sig  = ob.signal.filter.bandpass(channels[name], 2, 30, df = 500)
        axes.plot(channels["timestamp"], sig)
        
        for idx, row in channels["events"].iterrows():
            axes.axvline(idx, color='r', linestyle='--')
            
    
def Make_Epochs_for_Channels(channels, ch_names,n_samples_back, n_samples_forth):
    events = channels["events"]
    
    electrode_slices = {}
    
    for name in ch_names:
        filtered = ob.signal.filter.bandpass(channels[name], 2, 30, df = 500)
        
        
        time_series =pd.Series(np.array(filtered, dtype = 'float32'), index = channels['timestamp'] )
        electrode_slices[name] = Make_Slices_Groups(time_series, events, n_samples_back, n_samples_forth)
    
    return electrode_slices
    
def Make_Slices_Groups(data, events, n_samples_back, n_samples_forth):
    """Creates a dict of epochs (np.arrays) per event type
    
    Parameters
    ----------
    data: np.array
        whole EEG signal
    events: DataFrame
        Timestamps and description of events
    n_samples_back, n_samples_forth: (dict, dict)
        Key is event name, item is window size around time of event
    
    Returns
    -------
    grouped slices: dict
        Keys are event names, items are epochs (np.arrays)
        
    """

    grouped_slices = {}   
    
    for name, events_single_type in events.groupby('code'):
        #Specify useful events by storing them in a dicts specifying n_samples back and forth
        if(name in n_samples_back):
            grouped_slices[str(name)] = Cut_Slices_From_Signal(data, events_single_type, n_samples_back, n_samples_forth)    
    return grouped_slices
    
def Cut_Slices_From_Signal(_signal, events, n_samples_back, n_samples_forth):
#Iterates through all events and finds the index of the closest EEG sample in the _signal. 
#Takes a window spanning back and forth around the closest sample. 
    #Since it is already after groupby events ahve to have the same name at all indexes
    e_name = events['code'].iloc[0]
    slices = np.zeros((len(events.index), n_samples_back[e_name] + n_samples_forth[e_name]))    
    for idx, timestamp in enumerate(events.index):
        #Finding first index greater than event timestamp produces more error than finding the closest data timestamp regrdless of whether it is larger or smaller than event.
        event_index = Find_Closest_Sample(_signal,timestamp)
        #TODO check how inclusion/exclusion of lower and iupper bounds might affect the seslected slice size
        slices[idx,:] = _signal.iloc[event_index - n_samples_back[e_name] : event_index + n_samples_forth[e_name]]
    
    return slices
    
def Find_Closest_Sample(df, dtObj):
    return np.argmin(np.abs(df.index - dtObj))