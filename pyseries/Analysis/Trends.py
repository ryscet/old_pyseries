# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:19:24 2016

@author: ryszardcetnarski
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import pyseries.Analysis.Normalize
#import pyseries.Analysis.Normalize as norm

def Load_rs_envelopes():
    with open('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/sygnal_rest/processed/rest_envelopes.pickle', 'rb') as handle:
        b = pickle.load(handle)
    return b


def Load_rs_specgrams():
    with open('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/sygnal_rest/processed/rest_specgrams.pickle', 'rb') as handle:
        b = pickle.load(handle)
    return b

def Specgram_Trends():
    specgram  = Load_rs_specgrams()

    electrode_avg = {'F3':{},'F4':{}, 'P3':{}, 'P4':{}}
    for electrode in ['F3', 'F4', 'P3', 'P4']:
        bands_avg = {'all_spec':[], 'alpha':[], 'beta1':[], 'beta2':[], 'beta3':[]}
        for band in ['all_spec', 'alpha', 'beta1', 'beta2', 'beta3']:

            for subject_name in specgram.keys():
                band_val = specgram[subject_name][electrode][band]
                if(len(band_val) > 330):
                    bands_avg[band].append(band_val[0:330])
                else:
                    print(subject_name + ' too short')

        for band_name, band_value in bands_avg.items():
            bands_avg[band_name] = np.array(bands_avg[band_name])
        electrode_avg[electrode] = bands_avg

    for e_name, e_bands in electrode_avg.items():
        f, axes = plt.subplots(1,5)
        f.suptitle(e_name)
        i = 0


        for b_name, b_val in e_bands.items():
            b_val = np.log(b_val)
            for row in b_val:
                axes[0,i].plot(row, alpha = 0.7)
            axes[0,i].plot(np.mean(b_val, axis = 0), c = 'r')
            axes[0,i].set_ylabel(b_name)

            chunks=np.hsplit(b_val, 5)
            avgs = []
            stds = []
            for chunk in chunks:
                avgs.append(np.mean(chunk))
                stds.append(np.std(chunk))
            t = np.linspace(41.25, 288.75,5)

            axes[i].errorbar(t, avgs, yerr = stds, fmt = 'o')
            i = i+1



        f.tight_layout()
        path = '/Users/ryszardcetnarski/Desktop/Trends/'
        f.savefig(path + e_name +'_specgram_trends.png')




   # return electrode_avg


def EnvelopeSpecgramCorrelation():
    specgram  = Load_rs_specgrams()
    envelope  = Load_rs_envelopes()

    for (spec_name, spec_data), (env_name, env_data)  in zip( specgram.items(), envelope.items()):

        for (spec_e_name, spec_e_bands), (env_e_name, env_e_bands) in zip(spec_data.items(), env_data.items()):
            print('spec: %i  env: %i  r: %f' %(len(spec_e_bands['alpha']), len(env_e_bands['alpha']), len(env_e_bands['alpha']) / len(spec_e_bands['alpha']) ))
            data = np.array(env_e_bands['alpha'])
            spec_len = len(spec_e_bands['alpha'])
            chunks=np.split(data[0: spec_len *128 ], spec_len)

            binned_avg = [np.mean(chunk) for chunk in chunks]
          #  plt.plot(norm.Z_score(electrode_bands['alpha']))
            #plt.figure()
            #plt.plot(binned_avg, spec_e_bands['alpha'], 'o')

            coeff, p = stats.pearsonr(norm.Z_score(np.array(binned_avg)) , norm.Z_score(spec_e_bands['alpha']))
            if(p < 0.05):
                print('FOUND %f '%coeff)
                plt.figure()
                plt.plot(norm.Z_score(np.array(binned_avg)), norm.Z_score(spec_e_bands['alpha']), 'o')


    return binned_avg

