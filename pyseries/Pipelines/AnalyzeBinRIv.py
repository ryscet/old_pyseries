# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:27:09 2016

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
from scipy import signal


path = '/Users/user/Desktop/nagrania_eeg/binriv/Kuba_14_06_16/'
recording = loading.Read_edf.Combine_EDF_XML(path,0,10)

f, Pxx_den = signal.welch(recording['EEG P4'], fs = 498, nperseg=512)

plt.figure()
plt.plot(f, Pxx_den)


#epochs_before_info = {"response_changed": [ 498*5, 0] }
#
#epochs_before = prep.Epochs.Make_Epochs_for_Channels(recording, ['EEG P4'], epochs_before_info)['EEG P4']
#
#epochs_after_info = {"response_changed": [0, 498*5] }
#
#epochs_after = prep.Epochs.Make_Epochs_for_Channels(recording, ['EEG P4'], epochs_after_info)['EEG P4']
#
#epochs = {}
#epochs['P4'] = {'before_switch':epochs_before['response_changed'], 'after_switch': epochs_after['response_changed']}
#
#power_density= analysis.Explore.PlotPowerSpectrum(epochs['P4'], exact_sr =498, mode = 'period', name = path, freq_min = 0, freq_max = 100)


#f, Pxx_den = signal.welch(event, exact_sr, nperseg=512)
