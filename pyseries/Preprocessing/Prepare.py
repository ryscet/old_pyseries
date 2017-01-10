"""
Common steps to take before analysis. Data cleaning, filtering, normalizing.
"""
import numpy as np
import obspy as ob

def Filter_all(recording, min_freq = 1, max_freq = 70, df = 500):

	for key, value in recording.items():
		if(key not in ['events', 'timestamp']):
			recording[key] = ob.signal.filter.bandpass(value, min_freq, max_freq, df = sr)
	return recording


def Z_score_all(recording):

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

	for key, value in recording.items():
		if(key not in ['events', 'timestamp']):
			recording[key] = Z_score(value)
	return recording

def forward_difference(recording):
	for key, value in recording.items():
		if(key not in ['events', 'timestamp']):
			recording[key] = np.diff(value)
	#Forward difference does not exist for last entry. For a time stamp we take the time of the end of the difference
	recording['timestamp'] = recording['timestamp'][1::]
	return recording

def absolute_val(recording):
	for key, value in recording.items():
		if(key not in ['events', 'timestamp']):
			recording[key] = np.abs(value)
	#Forward difference does not exist for last entry. For a time stamp we take the time of the end of the difference
	#recording['timestamp'] = recording['timestamp'][1::]
	return recording


