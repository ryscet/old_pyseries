# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:07:48 2016

@author: user
"""

import sys
sys.path.insert(0, '/Users/user/Desktop/repo_for_pyseries/pyseries')

import pandas as pd
import numpy as np
import glob

import pyseries.LoadingData as loading
import pyseries.Preprocessing as prep
import pyseries.Analysis as analysis

all_rec = {}
paths = glob.glob('/Users/user/Desktop/nagrania_eeg/streams/*')
#paths = ['/Users/user/Desktop/nagrania_eeg/streams/Maciek_08_26_16',
#         '/Users/user/Desktop/nagrania_eeg/streams/Gosia_08_31_16']
for path in paths:
    #print(path)
    recording = loading.Read_edf.Combine_EDF_XML(path + '/', 3, 70)
    
    #Display markers over the whole signal    
    prep.Epochs.mark_events(recording,['EEG O1'], subject_name = path)
    
    #Define epochs for analysis
    epochs_info= {'ssvep_started': [500, 500*7], 'waiting_started' : [0, 500 *3] , "only_fixation_started" : [0, 500*2] }
    #Create them by slicing the signal
    epochs = prep.Epochs.Make_Epochs_for_Channels(recording, ['EEG O1','EEG O2','EEG P3', 'EEG P4'],epochs_info)
    #Re-reference, because only by subtracting P from O-electrodes ssvep response becomes visible
    new_ref = {}
    new_ref['ssvep'] = epochs ['EEG O2']['ssvep_started'] - epochs['EEG P4']['ssvep_started']
    #new_ref['waiting'] = epochs ['EEG O2']['waiting_started'] - epochs['EEG P4']['waiting_started']
    #new_ref['fixation'] = epochs ['EEG O2']["only_fixation_started"] - epochs['EEG P4']["only_fixation_started"]
    
    new_epochs  = {"O-P":new_ref}
    
    
    #Get the power spectra in two conditions       
    power_density= analysis.Explore.PlotPowerSpectrum(new_epochs['O-P'], 500, mode = 'period', name = path + " O2-P4", freq_min = 0, freq_max = 30)
   # power_density= analysis.Explore.PlotPowerSpectrum(epochs['EEG O2'], 500, mode = 'welch', name = "O2" + path)
  #  power_density= analysis.Explore.PlotPowerSpectrum(epochs['EEG P3'], 500, mode = 'welch', name = "P3" + path)
    #power_density= analysis.Explore.PlotPowerSpectrum(epochs['EEG P4'], 498, mode = 'period', name = "P4")
    
    
    all_rec[path] = recording
    
    
    
    
    
    












#
#
#log = pd.read_csv('/Users/user/Desktop/nagrania_eeg/streams/Aleksandra_07_15_16/unity_erp.csv', header = 1)
#
##Find indexes where miss was recorded
#misses = np.array(log[log['event'] == "miss"].index.tolist())
##find indexes where a trial ended and subtract 1 from them (i.e. get the indexes of the first row before trial ended)
#ends = np.array(log[log['event'] == "ssvep_ended"].index.tolist()) -1
##Find where two lists of indexes are the same, i.e. where a miss was recorded one row before trial ended
#inter = np.in1d(misses, ends)
##remove the intersecting indexes from all missess
#filtered_indexes = misses[~inter]
##make a copy of the original log
#log2 = log.copy()
##shift the miss event one row up ()
#log2.loc[filtered_indexes -1 , 'event'] ="miss"
##write a "target_appeared" in their old place
#log2.loc[filtered_indexes,  'event']="target_appeared" 
#
#log2.to_csv('/Users/user/Desktop/nagrania_eeg/streams/Aleksandra_07_15_16/unity_erp2.csv', index = False)