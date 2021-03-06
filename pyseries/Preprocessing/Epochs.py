# -*- coding: utf-8 -*-
"""
Epochs
======
Prepares a continous signal for analysis by cutting it into epochs around a specific event in the experiment. 
All types of events, with their corresponding timestamps, are saved in 'events' DataFrame inside the recording dict.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def Split_Epochs(recording, time_column, category_column,  back, forth):
    """ Epoch the signal around an event and divide the epochs between different conditions. 
        For example, select epochs for subject response and divide them between match and mismatch conditions.
    ----------
    recording: dict
        contains EEG channels, info, events and timestamps
    time_column: str
        Column name in 'events' with timestamps to base the epochs on. 
    category_column: str
        name of the column in the 'events' with the categorical variable to divide the epochs between different conditions. 
    back, forth: int, int
        size of the epoch in the number of samples going back and forth from the timestamp of the event. 
    
    Returns
    -------
    epochs: dict
        Epoched signal organized in a dictionary.
        Keys are event names, values are epochs (np.arrays)
    """
    
    # Prepare events by selecting a column with times of an event and a category column, when it is divided for some comparison, ex. match vs missmatch trials
    events = prepare_events(recording, time_column, category_column)
    data_matrix = np.array([recording[key] for key in recording['eeg_names']])
    all_slices = {code: [] for code in events['code'].unique()}
    
    for idx, event in events.iterrows():
        event_index = np.argmin(np.abs(recording['timestamp'] - np.datetime64(idx)))
        all_slices[event['code']].append(data_matrix[:, event_index - back : event_index + forth])
    
    return all_slices 

def Different_Epochs(recording, time_column_A, time_column_B,  back, forth):
    """ Epoch the signal around two different events.
        For example, select epochs for the delay period and for the response period.
    ----------
    recording: dict
        contains EEG channels, info, events and timestamps
    time_column_A, time_column_B: str, str
        Column names in 'events' with timestamps to base the epochs on. 
    back, forth: int, int
        size of the epoch in the number of samples going back and forth from the timestamp of the event. 
    
    Returns
    -------
    epochs: dict
        Epoched signal organized in a dictionary.
        Keys are event names, values are epochs (np.arrays)
    """
    
    # Prepare events by selecting a column with times of an event and a category column, when it is divided for some comparison, ex. match vs missmatch trials
    #events = prepare_events(recording, time_column, category_column)
    data_matrix = np.array([recording[key] for key in recording['eeg_names']])
    all_slices = {event: [] for event in [time_column_A, time_column_B]}
    print([time_column_A, time_column_B])
    for time_col in [time_column_A, time_column_B]:
        for event in recording['events'][time_col]:
            event_index = np.argmin(np.abs(recording['timestamp'] - np.datetime64(event)))
            all_slices[time_col].append(data_matrix[:, event_index - back : event_index + forth])

    return all_slices 

def prepare_events(recording, time_column, category_column):
    """Parse the events DataFrame with all experiment information into a smaller Dataframe with only relevant information for selected event type."""
    new_events = recording['events'][[time_column, category_column]]
    new_events.columns = ['time', 'code']
    new_events['time'] = pd.to_datetime(new_events['time'])
    new_events.set_index('time', inplace = True)

    recording[time_column] = new_events
    
    return new_events


def mark_events(recording, ch_names = []):
    """Plots raw signal with event markers on top.
    
    Parameters
    ----------
    recording: dict
        contains EEG channels, events and timestamps
    ch_names: list(str)
        List of channels to create epochs from
    """
    events = recording['events']
    #get number of event types
    unique_events =events['code'].unique()
    
    subject_name = recording['subject_name']
    
    # If the user does not specify otherwise, plot all eeg channels overlapping
    if not ch_names:
        ch_names = recording['eeg_names']
    
    #Choose colormap
    colormap = plt.cm.Paired # gist_ncar #nipy_spectral, Set1,Paired   
    colors = [colormap(i) for i in np.linspace(0, 1,len(unique_events))]
    
    color_dict =  {name: colors[i] for i, name in enumerate(unique_events)}
    
    fig, axes = plt.subplots(1,1)
    fig.suptitle(subject_name, fontweight = 'bold')
         
    for electrode_name in ch_names:

        
        sig = recording[electrode_name]
        axes.plot(recording["timestamp"], sig, alpha = 0.5, label = electrode_name)
        
        
        ymin, ymax = axes.get_ylim()
         #Pos for annotations
        ypos = ymax - np.abs(ymin) + np.abs(ymax)/3.0
        
    for idx, row in recording["events"].iterrows():
        axes.axvline(idx, linestyle='--',label = row['code'], color = color_dict[row['code']])
        axes.annotate(row['code'], xy = (idx, ypos), rotation = 90, color = color_dict[row['code']], horizontalalignment='right')            
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())



### OLD CODE BELOW ###
#
#def Make_Epochs_for_Channels(recording, ch_names, events_name, epoch_window):
#    """Epochs are pieces of signal which happened right after and/or before an event.
#    
#       `epochs_info` defines type of events and size of the epoch. 
#       `ch_names` defines which channles to make epochs from.
#       
#    
#    Parameters
#    ----------
#    recording: dict
#        contains EEG channels, events and timestamps
#    ch_names: list(str)
#        List of channels to create epochs from
#    events_name: str
#        name of the field in the recording dict with events prepared for epochs extraction
#    epochs_info: dict
#        Key is event name, value is an array [n_samples_back, n_samples_before]
#    
#    Returns
#    -------
#    epochs: dict
#        Epoched signal organized in a dictionary.
#        Keys are event names, values are epochs (np.arrays)
#        
#        
#    Examples
#    --------
#    >>> import pyseries.LoadingData as loading
#    >>> import pyseries.Preprocessing as prep
#    
#    First load the complete experiment recording, it will contain all EEG chanels, events log and timestamps:
#    
#    >>> recording = loading.Read_edf.Combine_EDF_XML('/Users/user/Desktop/Nagrania/rest/Rysiek_03_06/',True)
#    
#    Ten specify which events from the events log are to be used for epochs and their time from the event.
#    Epochs will be created only from events which name match those provided by keys in epochs info dictionary.
#    Values in epochs infor dictionary are [n_samples_before, n_samples_forward]         
#    
#    >>> epochs_info = {'Eyes Open': [0, 500 * 140], 'Eyes Closed': [0, 500 *140]}         
#    
#    Now you can make epochs:
#
#    >>> epochs= prep.Epochs.Make_Epochs_for_Channels(recording, ['EEG O1', 'EEG O2',  'EEG P3', 'EEG P4'],epochs_info)         
#        
#    """    
#    
#    events = recording[events_name]
#    
#    epochs = {}
#    
#    for name in ch_names:        
#        print(name)
#        time_series =pd.Series(np.array(recording[name], dtype = 'float32'), index = recording['timestamp'] )
#        epochs[name] = Make_Slices_Groups(time_series, events, epoch_window)
#    
#    return epochs
#    
#def Make_Slices_Groups(data, events, epoch_window, format = 'long'):
#    #Specify useful events by storing them in a dict specifying n_samples back and forth
#    # format of the data frame can be long (two columns, one with code another with time) or wide, then we pass a list of data frames
#    # Todo, implement wide format function. Do it.
#    grouped_slices = {}   
#    
#    for event_name, events_single_type in events.groupby('code'):
#        print(event_name)
#        #assert event_name in epochs_info.keys(), 'code not matching epochs name'
#        grouped_slices[str(event_name)] = Cut_Slices_From_Signal(data, events_single_type, epoch_window[event_name])    
#    return grouped_slices
#    
#
#
#def Cut_Slices_From_Signal(_signal, events, epoch_info):
##Iterates through all events and finds the index of the closest EEG sample in the _signal. 
##Takes a window spanning back and forth around the closest sample. 
#    slices = np.zeros((len(events.index), epoch_info[0] + epoch_info[1]))    
#    for idx, timestamp in enumerate(events.index):
#        #Finding first index greater than event timestamp produces more error than finding the closest data timestamp regrdless of whether it is larger or smaller than event.
#        try:
#            assert timestamp > _signal.index[0] and timestamp < _signal.index[-1], 'event %i not in range'%idx
#            event_index = Find_Closest_Sample(_signal,timestamp)
#            slices[idx,:] = _signal.iloc[event_index - epoch_info[0] : event_index + epoch_info[1]]
#        except AssertionError as e:
#            print(str(e))
#            pass
#    #Filter out 0 rows which had not been in range of assert
#    slices = slices[~np.all(slices == 0, axis=1)]
#    return slices

#
#from bokeh.charts import Line, show, output_file
#
#from bokeh.plotting import figure, output_file, show
#
#eeg_subset = ['EEG O1', 'EEG O2']
#data = {k: recording.get(k, None) for k in eeg_subset}
#
## create a line chart where each column of measures receives a unique color and dash style
#line = Line(data, y=eeg_subset,
#            dash=eeg_subset,
#            color=eeg_subset,
#            legend_sort_field = 'color',
#            legend_sort_direction = 'ascending',
#            title="Interpreter Sample Data", ylabel='Duration', legend=True)
#
#output_file("line_single.html", title="line_single.py example")
#show(line)

#
#def Find_Closest_Sample(signal, event_time):
#    #Subtract event time from an array containing all signal timestamps. Then the index at which the result of this subtraciton will be closest to 0 will be an index of event in the signal
#    return np.argmin(np.abs(signal.index - event_time))