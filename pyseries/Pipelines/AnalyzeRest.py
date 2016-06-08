# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:32:36 2016

@author: user
"""
import sys
sys.path.insert(0, '/Users/user/Desktop/repo_for_pyseries/pyseries/pyseries')

import pyseries.LoadingData as loading
import pyseries.Preprocessing as prep
import pyseries.Analysis as analysis




channels = loading.Read_edf.Combine_EDF_XML('/Users/user/Desktop/Nagrania/rest/Agnieszka_03_06/',True)

slices = prep.MakeSlices.Make_Slices_for_Channel(channels, ['EEG O1'],0, 500* 140)

prep.MakeSlices.mark_events(channels, ['EEG 01'])


event = analysis.Explore.PlotPowerSpectrum(slices['EEG O1'])


  


      


