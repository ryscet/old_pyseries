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
from io import open

import pandas as pd
import xml.etree.ElementTree as etree

import pyedflib 
import numpy as np
from datetime import datetime
import struct
import mne
import pandas as pd
import scipy.stats.mstats as sp
import glob

def MNE_Read_EDF(path):
    """Read .edf exported from digitrack using MNE library.
      
      Parameters
      ----------
      path:str
          directory of a folder containing the following files: sygnal.edf, unity_log.csv, digi_log.xml, digi_binary.1
          This is usually the folder with the subject name. ex: '/Users/rcetnarski/Desktop/Dane EEG/Pilot_Gabory/Maciek/experiment/'
  
      Returns
      -------
      raw_mne: mne.Raw object
          http://martinos.org/mne/dev/generated/mne.io.RawArray.html#mne.io.RawArray
          
      events: ndarray (n_events, 3)
          the first column is the sample number when index occured,
          second column is a mne requirement and has to be ignored,
          third column is the event code
          
      event_id: dict(event label : event code)
          dictionary with labels describing the event codes
    """
    # Load the eeg data  
    print(glob.glob(path +'*.edf'))
    paths = str(glob.glob(path +'*.edf'))
    assert len(glob.glob(path +'*.edf')) == 1,path + paths # Only one edf in the directory
    edf_path = glob.glob(path +'*.edf')[0] # from the sygnal.edf file
    
    raw_mne =  mne.io.read_raw_edf(edf_path,stim_channel = None, preload = True)
    # Fix the sampling rate info saved in the original sygnal.edf. Our software does not save it with high precision in the .edf file, so we will replace it manually.
    exact_sr = Get_Exact_Sampling_rate(path) # Get the high precision sampling rate
    raw_mne.info.update({'sfreq' : exact_sr }) # Update the mne info with the high precision sampling rate
    
    # Create a timestamp vector so event times can be expressed in sample number of eeg
    timestamp = exact_timestamp(path, raw_mne.n_times, exact_sr)
    # Read the events file
    log = pd.read_csv(path + 'unity_log.csv',parse_dates = True, index_col = 0, skiprows = 1, skipfooter = 1, engine='python')
    
    # Select the columns where the timestamp of the event was written
    event_time_columns =[col_name for col_name in log.columns if 'time' in col_name and 'response' not in col_name and 'psychopy' not in col_name and 'start_time' not in col_name]
    
    # Convert the timestamp in datetime format to a sample number counted from the first sample of EEG recording    
    # Find the index of event time in the timestamp vector. This index is the sample number of the event, relative to the first eeg sample.
    event_sample_indexes = {}
    # Iterate over columns with times for different events
    for time_col in event_time_columns:
        event_sample_indexes[time_col] = []
        print(time_col)
        # Iterate over each event
        for event in log[time_col]:
            # Nulls should only happen when we have a time that only appears in one condition. Example: response time does not appear in control condition
            if pd.notnull(event):
                #print(timestamp[10])
                #print(np.datetime64(event))
                event_index = np.argmin(np.abs(timestamp - np.datetime64(event)))
                event_sample_indexes[time_col].append(event_index)
    # Store the code and ints label in the the dictionary
    # IMPORTANT event code cannot be zero - the stimulus channel default value for no event is zero 
    event_id = {event_name : idx + 1 for idx, event_name in enumerate(event_time_columns)}

    # Process the events info untill it is in the format specified by MNE, i.e. ndarray with 3 columns
    events = pd.DataFrame(columns = ['sample_nr',  'code'])
    # Stack vertically all sample numbers for different events
    for event_label, sample_numbers in event_sample_indexes.items():
        tmp = pd.DataFrame(sample_numbers, columns = ['sample_nr'])
        tmp['code'] = event_id[event_label]
        # stack 
        events =  events.append(tmp)
   # Sort events chronologically
    events = events.sort_values(by = 'sample_nr')
   # Change to numpy array of ints
    events = events.as_matrix().astype('int')
   # MNE needs an extra column of zeros in the middle, it won't be used but has to be there
    events = np.insert(events, 1, 0, axis=1)


    return raw_mne, events, event_id, log


def Combine_EDF_XML(path):
    """Creates a dictionary with eeg signals, timestamps and events. 
       Reads edf file with eeg signal. Uses xml file to add timestamps to eeg. Reads unity_log with experiment events times. 
         
       Parameters
       ----------
       path:str
           directory containing .edf, .xml, and .csv files.

         Returns
         -------
         signal_dict (dict of Objects): dict
             stores eeg channels, ttimestamps and events.
             Keys:
             "EEG <channel name>" : eeg signal
             "timestamp" : timestamps for eeg
             "events" : names and timestamps of events
    
    """
    
    #---EEG SIGNAL PART---
    
    #Load the edf from the edf file generated by digitrack

    assert len(glob.glob(path +'*.edf')) == 1 # Only one edf in the directory
    signal_dict = Read_EDF(glob.glob(path +'*.edf')[0]) # from the sygnal.edf file
    # Store each eeg channel as an entry in a dictionary, key is electrode location, value is the signal.
    for chan_name, sig in signal_dict.items():
        signal_dict[chan_name] = sig
    
    #---EVENT MARKERS PART---
    print(path)
    #Read the event markers from the experiment and store in the same dictionary as signals under key 'events' 
    log = pd.read_csv(path + 'unity_log.csv',parse_dates = True, index_col = 0, skiprows = 1, skipfooter = 1, engine='python')
    signal_dict['events'] = log

    #---TIMESTAMP PART---
    #Get the timestamp based on the info from the exact_timestamp field in the .1 file  

    #Number of EEG samples
    signal_dict['n_samples'] = next(len(value) for (key,value) in signal_dict.items() if 'EEG' in key)

    #Sampling rate from the digi_binary.1 file
    signal_dict['sr'] = Get_Exact_Sampling_rate(path)

    #Expand the timestamp using the info about first sample time (from digi_log.xml) and sampling rate
    signal_dict['timestamp'] = exact_timestamp(path, signal_dict['n_samples'], signal_dict['sr'])
    
    #Add a list of all electrode names, usefull for plotting functions to loop over all electrodes
    signal_dict['eeg_names'] = [key for key in signal_dict.keys() if 'EEG' in key]
    
    #Keep tract of subject name, also useful for plotting
    signal_dict['subject_name'] = path.split('/')[-3]
    
    #Save the time of first sample for exporting data to edf
    signal_dict['first_sample_time'] = signal_dict['timestamp'][0]

    #store the timestamp in ms from start of eeg recording, for edf
    signal_dict['first_sample_time'] = signal_dict['timestamp'][0]
    
    signal_dict['timestamp_ms'] = (signal_dict['timestamp'] - signal_dict['first_sample_time']).astype('timedelta64[ms]').astype('float')

    return signal_dict
 
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
    #print('Channels:')
    for idx, name in enumerate(signal_labels):
        #print(name.decode("utf-8"))
        signal_dict[name.decode("utf-8")] = f.readSignal(idx)
        
    f._close()

    return signal_dict
    
        
    
def exact_timestamp(path, n_samples, sampling_rate):
    """Elmiko EEG amplifier 1042 calibrates to produce a sampling rate defined by the user. 
        The calibrated sampling rate is slightly different from user defintion.
        Calibrated sampling rate is saved in the header of digi_binary.1 file.
        """
    #Convert to nanoseconds by multiplying to desired resolution and cutting the reminding decimal places using int(). *time units change by order of 10^3
    #Conversion will be used to produce a high precision timestamp (something went funny in the np.timedelta64 function when trying to use ms instead of ns)
    #NOTE: exact_sr_ns is actually a sample duration in nanoseconds
    exact_sr_ns = int(1000.0/sampling_rate*10**3 *10**3)
    # Create an empty time vector of nanosecond precision datetimes
    timestamp = np.empty(n_samples, dtype='datetime64[ns]')
    # Set the first value using the first sample time saved by digi track
    timestamp[0] =Read_XML(path)['DateTime'].iloc[0]
    # Populate the time vector by adding sample duration to the next sample.
    for i in range(n_samples - 1):
        timestamp[i+1] = timestamp[i] + np.timedelta64(exact_sr_ns, 'ns')

    return timestamp

def Read_XML(path):
    """Read the header for the signal from .EVX.

      Returns
      -------
      df: DataFrame
          Contains timestamp marking first EEG sample
    """
    
    assert len(glob.glob(path +'*.evx')) == 1 # Only one digi log file in the directory
    
    with open(glob.glob(path +'*.evx')[0], mode='r',encoding='utf-8') as xml_file:
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
#HACK changing timezone by manually adding one hour
#TODO make sure the timestamps will be possible to comapre between tz (utc) naive and tz aware formats
            timezone_info = dt_time.find('+') # There an offset from some annoying timezone is saved
            df['UNIXTIME'] = pd.to_datetime([u_time], unit='us').tz_localize('UTC') + pd.Timedelta(hours = int(dt_time[timezone_info +1: dt_time.find('+')+3]))
            df['DateTime'] = pd.to_datetime([dt_time],infer_datetime_format =True).tz_localize('UTC')+ pd.Timedelta(hours = int(dt_time[timezone_info +1: dt_time.find('+')+3]))
    return df

        

def Get_Exact_Sampling_rate(path):
    #Read the bytes from .1 file

    assert len(glob.glob(path +'*.1')) == 1 # Only one digi binary file in the directory
    
    with open(glob.glob(path +'*.1')[0], "rb") as binary_file:
        
         #Seek position and read N bytes
        binary_file.seek((490+(89*64)))  # Go to bite nr
        couple_bytes = binary_file.read(8)
        sr = struct.unpack("d", couple_bytes)
    
    print('!!!!!!!!!!!! REMEMBER EXG ETHERNET DIFFERENCE FOR SR !!!!!!!!!!!!!!')   

    assert sr[0] > 100 and sr[0] < 10000
    
    return sr[0]
    
