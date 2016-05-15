# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:21:47 2016

@author: ryszardcetnarski
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats


def Load_rs_envelopes():
    with open('/Users/user/Desktop/Data/sygnal_rest/processed/rest_envelopes.pickle', 'rb') as handle:
        b = pickle.load(handle)
    return b


def Load_rs_specgrams():
    with open('/Users/user/Desktop/Data/sygnal_rest/processed/rest_specgrams.pickle', 'rb') as handle:
        b = pickle.load(handle)
    return b




def BetweenBandsCorr(band_A, band_B, dtype, lag_size, n_bins):

    if(dtype == 'specgram'):
        data = Load_rs_specgrams()

    if(dtype == 'envelope'):
        data = Load_rs_envelopes()

    all_electrode_coefs = {'F3':[], 'F4':[], 'P3':[], 'P4':[]}
    all_electrode_p= {'F3':[], 'F4':[], 'P3':[], 'P4':[]}

    for subject_name, subject_data in data.items():
        for electrode_name, electrode_bands in subject_data.items():
            all_coeffs, all_p, lags = CrossCorrelate(electrode_bands[band_A],electrode_bands[band_B], lag_size, n_bins)

            all_electrode_coefs[electrode_name].append(all_coeffs)
            all_electrode_p[electrode_name].append(all_p)


    f, axes = plt.subplots(4,2)
    f.suptitle(band_A + ' cross-corr ' +band_B)
    electrode_idx = 0

    for (e_name, electrode_coeff), (e_name2, electrode_p) in zip(all_electrode_coefs.items(), all_electrode_p.items()):

        all_e_coeffs = []
        all_e_p = []

        for subject_coeffs, subject_p in zip(electrode_coeff, electrode_p):

            axes[electrode_idx, 0].plot(lags, subject_coeffs, alpha = 0.2)
            axes[electrode_idx, 1].plot(lags, subject_p, alpha = 0.2)

            all_e_coeffs.append(subject_coeffs)
            all_e_p.append(subject_p)


        coeff_mean =  np.mean(np.array(all_e_coeffs), axis= 0)
        p_mean =  np.mean(np.array(all_e_p), axis= 0)

        axes[electrode_idx, 0].plot(lags, coeff_mean, c = 'r')
        axes[electrode_idx, 1].plot(lags, p_mean, c = 'r')


        axes[0, 0].set_title('corr coeff')
        axes[0, 1].set_title('p value')

        axes[0, 0].set_ylabel('F3')
        axes[1, 0].set_ylabel('F4')
        axes[2, 0].set_ylabel('P3')
        axes[3, 0].set_ylabel('P4')

        electrode_idx = electrode_idx+1

    f.tight_layout()
    path = '/Users/ryszardcetnarski/Desktop/Cross_correlations/'
    #f.savefig(path + dtype +'_' + band_A +'_'+band_B+'.png')


    #return all_e_coeffs



def CrossCorrelate(x,y, absolute_lag, binsize):
    #x = np.sin(np.arange(0, 1440 * pi/180, 0.1) + 45* pi/180 )
    #y = np.sin(np.arange(0, 1440 * pi/180, 0.1))

    #f, axes = plt.subplots(2)
    #axes[0].plot(x,'r')
    #axes[0].plot(y, 'b')

  #  print(len(x))

    lags = np.arange(-absolute_lag, absolute_lag+1,  binsize, dtype = 'int')
   # print(len(lags))
    all_coeffs = []
    all_p = []
    for lag in lags:
  #      print(lag)
        if(lag <= 0):
            tmp_x = x[ 0 : len(x) + lag]
            tmp_y = y[0 + abs(lag) ::]
        if(lag > 0 ):
            tmp_x = x[0 + abs(lag) ::]
            tmp_y = y[0 : len(x) - lag]

    #    print(len(tmp_x))
     #   print(len(tmp_y))

        coeff, p = stats.pearsonr(tmp_x , tmp_y)
        all_coeffs.append(coeff)
        all_p.append(p)

    #axes[1].plot(lags,all_coeffs)
    return all_coeffs,all_p, lags

