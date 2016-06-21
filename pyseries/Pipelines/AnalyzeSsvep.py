# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:21:35 2016

@author: user
"""
import sys
sys.path.insert(0, '/Users/user/Desktop/repo_for_pyseries/pyseries')

import pyseries.LoadingData as loading
import pyseries.Preprocessing as prep
import pyseries.Analysis as analysis
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


#path = '/Users/user/Desktop/eeg/ssvep/Blazej 13.06.16/'
#epochs = loading.Read_edf.Combine_EDF_XML(path,True)
#prep.Epochs.mark_events(epochs,['EEG O2'] )


def calc_corr(path):
    epochs = loading.Read_edf.Combine_EDF_XML(path,True)
    
    #prep.Epochs.mark_events(epochs, ['S1', 'S2'])
    
    #
    
    
    epochs_info= {"Please Count": [0, 500*10], "Only Look": [0, 500 *10]}
    
    
    
    slices = prep.Epochs.Make_Epochs_for_Channels(epochs, ['EEG O1','EEG O2','EEG P3', 'EEG P4',  ],epochs_info)
    
    new_ref = {}
    new_ref['Only Look'] = slices['EEG O1']['Only Look'] - slices['EEG P3']['Only Look']
    new_ref['Please Count'] = slices['EEG O1']['Please Count'] - slices['EEG P3']['Please Count']
    new_slices = {"O-P":new_ref}
    
    
    responses = epochs['events'][epochs['events']["code"] == "responded"]
    accuracy = responses['tmp'] / responses['tmp2']
    
    
    
    f, pxx = analysis.Explore.PlotPowerSpectrum(new_slices['O-P'], 498, 'period')
    
    ssvep = pxx[:,50]
    ssvep = ssvep /max(ssvep)
    accuracy = accuracy /max(accuracy)
    
    return ssvep, accuracy, pxx
    

def plot_corr():
    
    paths = ['/Users/user/Desktop/eeg/ssvep/Blazej 13.06.16/',
            '/Users/user/Desktop/eeg/ssvep/Ania_14_06_16/',
            '/Users/user/Desktop/eeg/ssvep/Karen_14_06_16/',
            '/Users/user/Desktop/Nagrania/ssvep_count/Agnieszka_03_06/'
            ]

    all_ssvep = []
    all_acc = []
    for p in paths:
        ssvep, acc, pxx = calc_corr(p)
        all_ssvep.extend(ssvep)
        all_acc.extend(acc)
    
    plt.plot(all_ssvep, all_acc)    
    fig, axes = plt.subplots(1)
    print(r2(all_ssvep, all_acc))
    #sns.jointplot(x = all_ssvep,y =  all_acc, kind="reg")


def r2(x, y):
    return stats.pearsonr(x, y)
#
#
#  
