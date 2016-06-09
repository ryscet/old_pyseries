# -*- coding: utf-8 -*-
"""
ArtifactRemoval
===============
Initial filtering. Artifacts:
	* EOG
	* ECG
	* power line
"""


import obspy.signal.filter as ob


def band_pass(sig, min_freq, max_freq):
    """Plain bandpass

    Parameters
    ----------
    sig: np.array
        whole EEG signal (single channel)
    min_freq, max_freq: (int, int)
        lower and upper bound of bandpass filter.

    Returns
    -------
    filtered: nparray
    	filtered signal between min_freq, max_freq, default sampling rate is 500 hz

    """
    filtered = ob.bandpass(sig,min_freq, max_freq, df = 500)
    return filtered
#    ssvep_fast = loading.Read_edf.Combine_EDF_XML('/Users/user/Desktop/Nagrania/ssvep_count/Rysiek_03_06/')
 #   p3_ssvep = filters.bandpass(ssvep_fast['EEG P3'], 2, 30, df = 500)
    #plt.figure()
    #plt.plot(p3_ssvep)
    

