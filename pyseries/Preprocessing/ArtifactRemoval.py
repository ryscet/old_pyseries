# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:43:05 2016

@author: user
"""

import sys
sys.path.insert(0, '/Users/user/Desktop/repo_for_pyseries/pyseries/pyseries')

#import LoadingData as loading
#import Pipelines as pipes
import obspy.signal.filter as filters
#import matplotlib.pyplot as plt




def band_pass(sig, min_freq, max_freq):
    filtered = filters.bandpass(sig,min_freq, max_freq, df = 500)
    return filtered
#    ssvep_fast = loading.Read_edf.Combine_EDF_XML('/Users/user/Desktop/Nagrania/ssvep_count/Rysiek_03_06/')
 #   p3_ssvep = filters.bandpass(ssvep_fast['EEG P3'], 2, 30, df = 500)
    #plt.figure()
    #plt.plot(p3_ssvep)
    

