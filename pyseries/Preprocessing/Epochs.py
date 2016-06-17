# -*- coding: utf-8 -*-
"""
Epochs
======

Making epochs (i.e slicing the signal) around events. 
After epoching analysis can be performed, like erp's or spectrograms.
"""
import obspy as ob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Make_Epochs_for_Channels(recording, ch_names, epochs_info):
    """Epochs are pieces of signal which happened right after and/or before an event.
    
       `epochs_info` defines type of events and size of the epoch. 
       `ch_names` defines which channles to make epochs from.
       
    
    Parameters
    ----------
    recording: dict
        contains EEG channels, events and timestamps
    ch_names: list(str)
        List of channels to create epochs from
    epochs_info: dict
        Key is event name, value is an array [n_samples_back, n_samples_before]
    
    Returns
    -------
    grouped slices: dict
        Keys are event names, items are epochs (np.arrays)
        
        
    Examples
    --------
    >>> import pyseries.LoadingData as loading
    >>> import pyseries.Preprocessing as prep
    
    First load the complete experiment recording, it will contain all EEG chanels, events log and timestamps:
    
    >>> recording = loading.Read_edf.Combine_EDF_XML('/Users/user/Desktop/Nagrania/rest/Rysiek_03_06/',True)
    
    Ten specify which events from the events log are to be used for epochs and their time from the event.
    Epochs will be created only from events which name match those provided by keys in epochs info dictionary.
    Values in epochs infor dictionary are [n_samples_before, n_samples_forward]         
    
    >>> epochs_info = {'Eyes Open': [0: 500 * 140], 'Eyes Closed[0: 500 *140]'}         
    
    Now you can make epochs:

    >>> epochs= prep.Epochs.Make_Epochs_for_Channels(recording, ['EEG O1', 'EEG O2',  'EEG P3', 'EEG P4'],epochs_info)         
        
    """    
    
    events = recording["events"]
    
    electrode_slices = {}
    
    for name in ch_names:
        filtered = ob.signal.filter.bandpass(recording[name], 2, 30, df = 500)
        
        
        time_series =pd.Series(np.array(filtered, dtype = 'float32'), index = recording['timestamp'] )
        electrode_slices[name] = Make_Slices_Groups(time_series, events, epochs_info)
    
    return electrode_slices
    
def Make_Slices_Groups(data, events, epochs_info):
    #Specify useful events by storing them in a dict specifying n_samples back and forth

    grouped_slices = {}   
    
    for event_name, events_single_type in events.groupby('code'):
        if(event_name in epochs_info.keys()):
            grouped_slices[str(event_name)] = Cut_Slices_From_Signal(data, events_single_type, epochs_info[event_name])    
    return grouped_slices
    
def Cut_Slices_From_Signal(_signal, events, epoch_info):
#Iterates through all events and finds the index of the closest EEG sample in the _signal. 
#Takes a window spanning back and forth around the closest sample. 
    slices = np.zeros((len(events.index), epoch_info[0] + epoch_info[1]))    
    for idx, timestamp in enumerate(events.index):
        #Finding first index greater than event timestamp produces more error than finding the closest data timestamp regrdless of whether it is larger or smaller than event.
        event_index = Find_Closest_Sample(_signal,timestamp)
        slices[idx,:] = _signal.iloc[event_index - epoch_info[0] : event_index + epoch_info[1]]
    
    return slices

def mark_events(recording, ch_names):
    """Plots raw signal with event markers on top.
    
    Parameters
    ----------
    recording: dict
        contains EEG channels, events and timestamps
    ch_names: list(str)
        List of channels to create epochs from
    """
    
    for name in ch_names:
        fig, axes = plt.subplots(1,1)
        
        sig  = ob.signal.filter.bandpass(recording[name], 2, 30, df = 500)
        axes.plot(recording["timestamp"], sig)
        
        for idx, row in recording["events"].iterrows():
            axes.axvline(idx, color='r', linestyle='--')


def Find_Closest_Sample(signal, event_time):
    #Subtract event time from an array containing all signal timestamps. Then the index at which the result of this subtraciton will be closest to 0 will be an index of event in the signal
    return np.argmin(np.abs(signal.index - event_time))