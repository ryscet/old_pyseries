"""
Common steps to take before analysis. Data cleaning, filtering, normalizing.
"""
import numpy as np
import obspy as ob

def trim_outside_experiment(recording):
    #returning from this function does not make any sense as it operates by reference
    #warning the different type of time datatypes comparison is only possible in that order: Timestamp < numpy.datetime64
    assert recording['events'].index[0] > recording['timestamp'][0] and recording['events'].index[-1] < recording['timestamp'][-1], 'events not in range'
    
    
    start_index = np.argmin(np.abs(recording['timestamp'] - np.datetime64(recording['events'].index[0]))) - 2
    end_index = np.argmin(np.abs(recording['timestamp'] - np.datetime64(recording['events'].index[-1]))) + 2
        
    for ch_name in recording['eeg_names']:
        recording[ch_name] = recording[ch_name][start_index: end_index]

    recording['timestamp'] = recording['timestamp'][start_index: end_index]

    return recording


def Filter_all(recording, min_freq = 1, max_freq = 70):

	for key in recording['eeg_names']:
		recording[key] = ob.signal.filter.bandpass(recording[key], min_freq, max_freq, df = int(recording['sr']))
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


