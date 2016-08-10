# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:32:36 2016

@author: user
"""
import sys
sys.path.insert(0, '/Users/user/Desktop/repo_for_pyseries/pyseries/')

import pyseries.LoadingData as loading
import pyseries.Preprocessing as prep
import pyseries.Analysis as analysis


def plot_rest():
    paths = [   '/Users/user/Desktop/nagrania_eeg/rest/Ania_14_06_16/',
                '/Users/user/Desktop/nagrania_eeg/rest/Karen_14_06_16/',
                '/Users/user/Desktop/nagrania_eeg/rest/Agnieszka_03_06_16/',
                '/Users/user/Desktop/nagrania_eeg/rest/Kuba_14_06_16/',
                '/Users/user/Desktop/nagrania_eeg/rest/Rysiek_03_06_16/',
                '/Users/user/Desktop/nagrania_eeg/rest/Aleksandra_07_15_16/',
                '/Users/user/Desktop/nagrania_eeg/rest/Rysiek_07_21_16/',
                '/Users/user/Desktop/nagrania_eeg/rest/Aleksandra_07_21_16/',
                '/Users/user/Desktop/nagrania_eeg/rest/Agnieszka_07_21_16/'

                ]


    for idx, path in enumerate(paths):
        print(path)
        recording = loading.Read_edf.Combine_EDF_XML(path, 3, 70)
        
        epochs_info= {"Eyes Open": [0, 498*140], "Eyes Closed": [0, 498 *140]}
        
        epochs = prep.Epochs.Make_Epochs_for_Channels(recording, ['EEG O1'],epochs_info)
        
        power_density= analysis.Explore.PlotPowerSpectrum(epochs['EEG O1'], 498, mode = 'welch', name = path, save_path ="/Users/user/Desktop/Figures/rest/" + str(idx) + ".png"   )

        prep.Epochs.mark_events(recording,['EEG O1'], subject_name = path)
