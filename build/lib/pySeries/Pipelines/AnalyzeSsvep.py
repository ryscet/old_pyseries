# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:21:35 2016

@author: user
"""

#import sys
#sys.path.insert(0, '/Users/user/Desktop/repo_for_pyseries/pyseries/')

import pyseries.LoadingData as loading
import pyseries.Preprocessing as prep
import pyseries.Analysis as analysis

#channels = loading.Read_edf.Combine_EDF_XML('/Users/user/Desktop/Nagrania/ssvep_20hz/Agnieszka_03_06/',True)
#
#n_samples_back = {"iti_screen": 0, "ssvep_ended": 500 * 10}
#n_samples_forth = {"iti_screen":  500* 10, "ssvep_ended": 0}
#
#
#slices = prep.MakeSlices.Make_Slices_for_Channel(channels, ['EEG O1'],n_samples_back, n_samples_forth)
#
#prep.MakeSlices.mark_events(channels, ['EEG 01'])
#
#
#f, pxx = analysis.Explore.PlotPowerSpectrum(slices['EEG O1'])
#
#
#  
#
