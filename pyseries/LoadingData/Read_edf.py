# -*- coding: utf-8 -*-
"""
Read_edf
========

Reading data from Elmiko DigiTrack. Integrating time info from XML (.EVX file from digitrack) about time of first EEG sample
with sampling rate info (from .1 file from digitrack) to make timestamps for EEG signal. EEG signal needs to be exported to .edf 
from digitrack, then it can be parsed here.

Use timestamps from experiment log file to cut slices from EEG around events. EEG and events need to be saved with respect to the same 
clock, so best do experiment and recording on the same machine.
"""

import pandas as pd
#import xml.etree.ElementTree as etree
import pyedflib 
import numpy as np
from datetime import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import obspy.signal.filter as filters

##bandstop(data, freqmin, freqmax, df, corners=4, zerophase=False)[source]


#TODO Make an organized/relative paths way of maintaining database
#TODO make a way of marking event time on the spectrogram (approximate)
def Run():
    n_back =  500
    n_forth = 12
    
    electrode_slices = Make_Slices_for_Channel(['EEG Fpz-Cz'], n_back, n_forth )

    PlotErp(electrode_slices['EEG Fpz-Cz'], n_back)
    
    PlotSpectrogram(electrode_slices['EEG Fpz-Cz'], n_back, n_forth)
    
    PlotPowerSpectrum(electrode_slices['EEG Fpz-Cz'])

def Make_Slices_for_Channel(ch_names,n_samples_back, n_samples_forth):
    channels = Combine_EDF_XML()
    events = Random_Events()
    
    electrode_slices = {}
    
    for name in ch_names:
        ts =pd.Series(np.array(channels[name], dtype = 'float32'), index = channels['timestamp'] )
        electrode_slices[name] = Make_Slices_Groups(ts, events, n_samples_back, n_samples_forth)
    
    return electrode_slices

#electrode_slices = Make_Slices_for_Channel(['EEG Fpz-Cz'])


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

def CalcEventPosition(n_back, n_forth, n_fft_windows):
    pos = n_back / ( (n_back+n_forth) / n_fft_windows )
    #print(pos)
    return pos

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

def PlotPowerSpectrum(electrode_slices):
    sns.set()
    fig, axes = plt.subplots(1)
    for name, event in electrode_slices.items():
    
        f, Pxx_den = signal.welch(event, 100, nperseg=32)
        #avg_Pxx = np.mean(Pxx_den, axis = 0)        
        
        if('condtions_Pxx' not in locals()):
            condtions_Pxx = Pxx_den
        else:
            condtions_Pxx = np.dstack((condtions_Pxx , Pxx_den))       
    
    sns.tsplot(data=np.log(condtions_Pxx), time = f,  err_style="unit_traces", condition = [key for key in electrode_slices.keys()], ax = axes)
                     
    axes.set_yticklabels(labels = f, rotation = 0)
    
    axes.set_ylabel('Welch Power Density')
    axes.set_xlabel('frequency')



    
def AverageSpectrogram(electrode_event):

    f, t, Sxx = signal.spectrogram(electrode_event, window = 'hamming', fs = 100,nperseg = 64, noverlap = 32, return_onesided =True, scaling = 'spectrum' )
    
    Sxx = np.mean(np.log(Sxx), axis = 0)
    
    upper_bound = np.argmax( f > 13.0 )

    f = [format(x, '.1f') for x in f]
            

    return f, t, Sxx, upper_bound


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
    
    for name, events_single_type in events.groupby('event_code'):
        grouped_slices[str(name)] = Cut_Slices_From_Signal(data, events_single_type, n_samples_back, n_samples_forth)    
    return grouped_slices

def Combine_EDF_XML():
    """Extracts EEG channels data from edf and creates a new channel with timestamps. 
         
         Returns
         -------
         signal_dict: dict
             stores EEG timeseries and timestamps
    
    """
    signal_dict = Read_EDF()
    start_time = Read_XML()
    
    #freq = 1000ms / 500 i.e. how much time between each sample
    freq='2ms' 
#Assume the longest variable in the .edf must be EEG, so use this length for timestamps
    n_samples = GetMaxLength(signal_dict)
    print("n samples %i"%n_samples)

#TODO Could also use start_time['DateTime'] - check which is better
    index = pd.date_range(start_time['UNIXTIME'].iloc[0], periods= n_samples, freq = freq)
    
    signal_dict['timestamp'] = index
    return signal_dict
    
def GetMaxLength(_dict):        
    maks=max(_dict, key=lambda k: len(_dict[k]))
    return len(_dict[maks])



#path = '/Users/user/Desktop/Resty/Ewa_resting_state.edf'

def Read_EDF(path):
    """Read .edf exported from digitrack and converts them to a dictionary.
        
        Parameters
        ----------
        path:str
            directory of .edf
    
        Returns
        -------
        signal_dict: dict(np.array)
            Keys are channel names
    """
    
    
    f = pyedflib.EdfReader(path)
    #n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    
    
    signal_dict = {}
    for idx, name in enumerate(signal_labels):
        print(name.decode("utf-8"))
        signal_dict[name.decode("utf-8")] = f.readSignal(idx)
        
    f._close()

    return signal_dict


        

    
def Find_Closest_Sample(df, dtObj):
    return np.argmin(np.abs(df.index - dtObj))
    

def Read_XML():
    """Read the header for the signal from .EVX.

       Returns
       -------
       df: DataFrame
           Contains timestamp marking first EEG sample
    """
    
    path = '/Users/user/Desktop/ReadyDigiDataBase/test.xml'    
    with open(path, 'r') as xml_file:
        xml_tree = etree.parse(xml_file)        
        root = xml_tree.getroot()
#Get only the relevant fields  
    for child_of_root in root:
        if(child_of_root.attrib['strId'] == 'Technical_ExamStart'):
            time_event = child_of_root.find('event')
            #Timestamp in unix time
            u_time = time_event.attrib['time']
            #Timestamp in DateTime
            dt_time = time_event.find('info').attrib['time']
#store this information in a dataframe in a datetime/timestamp format
            df = pd.DataFrame()            
#HACK changing timezone by manually adding two hours
#TODO make sure the timestamps will be possible to comapre between tz (utc) naive and tz aware formats
            df['UNIXTIME'] = pd.to_datetime([u_time], unit='us') + pd.Timedelta(hours =2)
            df['DateTime'] = pd.to_datetime([dt_time], infer_datetime_format=True) + pd.Timedelta(hours =2)
            
    return df

    
def Random_Events():

    dates = []
    
    for i in range(0,20): 
        year = 2016
        month = 5
        day = 11
        hour = random.randint(13, 14)
        minute = random.randint(42,59)
        second = random.randint(0,59)
        microseconds = 102001
        dates.append(datetime(year, month, day, hour, minute, second, microseconds))
        
    df = pd.DataFrame(index =pd.to_datetime(dates, infer_datetime_format=True))
    df['event_code'] =  np.hstack((np.zeros(10), np.ones(10)))

    return df.sort_index()
    
def ReadEvents():
    log = pd.read_csv('/Users/user/Desktop/Resty/log.csv',parse_dates = True)
    timestamps = pd.to_datetime(log['time'])
    
    
#    
#def Sample_Data():
#
#    starting_time = Read_XML()['DateTime']
#    sig_len = 25000
#    signal = np.random.randn(sig_len)
##Check if last index matches the ending timestamp from the XML header
#    index = pd.date_range(starting_time.iloc[0] , periods=len(signal), freq='250ms')
#    ts = pd.Series(signal, index=index )
#    return ts