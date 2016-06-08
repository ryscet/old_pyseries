# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:32:36 2016

@author: user
"""
#import sys
#sys.path.insert(0, '/Users/user/Desktop/repo_for_pyseries/pyseries/')

import pyseries.Pipelines
import pyseries.LoadingData as loading
import pyseries.Preprocessing as prep
import pyseries.Analysis as analysis




#channels = loading.Read_edf.Combine_EDF_XML('/Users/user/Desktop/Nagrania/rest/Rysiek_03_06/',True)
#
#n_samples_back = {"Eyes Open": 0, "Eyes Closed": 0}
#n_samples_forth = {"Eyes Open":  500* 140, "Eyes Closed":  500* 140}
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
#
#      


