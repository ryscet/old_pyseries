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
import xml.etree.ElementTree as etree

import pyedflib 
import numpy as np
from datetime import datetime
import random
import struct
import pyseries.Preprocessing.ArtifactRemoval as ar




#TODO Make an organized/relative paths way of maintaining database
#path = '/Users/user/Desktop/Nagrania/rest/Rysiek_03_06/'
def Combine_EDF_XML(path, bandpass_filter):
    """Extracts EEG channels data from edf and creates a new channel with timestamps. 
         
         Returns
         -------
         signal_dict: dict
             stores EEG timeseries and timestamps
    
    """
    signal_dict = Read_EDF(path + "sygnal.edf")
    start_time = Read_XML(path + "digi_log.xml")

    if(bandpass_filter == True):
        for chan_name, sig in signal_dict.items():
            signal_dict[chan_name] = ar.band_pass(sig, 2,30)

    
    #freq = 1000ms / 500 i.e. how much time between each sample
    freq='2ms' 
#Assume the longest variable in the .edf must be EEG, so use this length for timestamps
    n_samples = GetMaxLength(signal_dict)
    print("n samples %i"%n_samples)

#TODO Could also use start_time['DateTime'] - check which is better
    index = pd.date_range(start_time['UNIXTIME'].iloc[0], periods= n_samples, freq = freq)
    
    signal_dict['timestamp'] = index


    log = pd.read_csv(path + 'unity_log.csv',parse_dates = True, index_col = 0, skiprows = 1, skipfooter = 1, engine='python')
    
    signal_dict['events'] = log

    e_ts = exact_timestamp(path, n_samples)
#TODO decide which timestamp is correct
    signal_dict['exact_timestamp'] = e_ts

    
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


        

    

    

def Read_XML(path):
#    import xml.etree.cElementTree as ET
    """Read the header for the signal from .EVX.

       Returns
       -------
       df: DataFrame
           Contains timestamp marking first EEG sample
    """
    
    
    with open(path, mode='r',encoding='utf-8') as xml_file:
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
            df['DateTime'] = pd.to_datetime([dt_time],infer_datetime_format =True) + pd.Timedelta(hours =2)
    return df

    

def exact_timestamp(path, n_samples):

    #1000.0/Get_Exact_Sampling_rate(path)*1000 *1000
   # freq = '2008147ns'
    timestamp = np.empty(n_samples, dtype='datetime64[ns]')
    timestamp[0] =Read_XML(path + 'digi_log.xml')['DateTime'].iloc[0]
    for i in range(n_samples - 1):
        timestamp[i+1] = timestamp[i] + np.timedelta64(2008147, 'ns')

    return timestamp
    

def Get_Exact_Sampling_rate(path):
    
    with open(path + 'digi_binary.1', "rb") as binary_file:
         #Seek position and read N bytes
        binary_file.seek((490+(89*32)))  # Go to bite nr
        couple_bytes = binary_file.read(8)
        sr = struct.unpack("d", couple_bytes)
        print(sr)
        
    return sr[0]
    

    
#    
#    
#def Random_Events():
#
#    dates = []
#    
#    for i in range(0,20): 
#        year = 2016
#        month = 5
#        day = 31
#        hour = random.randint(13, 14)
#        minute = random.randint(42,59)
#        second = random.randint(0,59)
#        microseconds = 102001
#        dates.append(datetime(year, month, day, hour, minute, second, microseconds))
#        
#    df = pd.DataFrame(index =pd.to_datetime(dates, infer_datetime_format=True))
#    df['code'] =  np.hstack((np.zeros(10), np.ones(10)))
#
#    return df.sort_index()    
    
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