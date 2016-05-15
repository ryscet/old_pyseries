# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:42:11 2016

@author: ryszardcetnarski
"""

import numpy as np

def Z_score(vector):
##Sanity check, should be a numpy array anyways. If not np, then subtraction might not subtract a constant from all elements
    vector = np.array(vector)
    copy = vector
    nan_idx = np.isnan(vector)
    #Compute zscore for non nans
    vector = vector[~nan_idx]
    z_score = (vector - np.mean(vector))/np.std(vector)
    #Substitute non nans with z score and keep the remaining nans
    copy[~nan_idx] = z_score
    return copy
