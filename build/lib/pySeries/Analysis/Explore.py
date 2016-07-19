"""
Explore
=======

Plots for common EEG analysis methods
    * Event related potentials (ERP's)
    * Periodogram and Welch power spectrum
    * Spectrogram 

Requires the signal to be epoched by Preprocessing.MakeSlices method.
Epochs are grouped and colored by conditions.
"""

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np



def PlotPowerSpectrum(electrode_slices, exact_sr =498, freq_min = 0, freq_max = 100, mode = 'period', name = ''):
    """Plot average and individual traces of power spectrum for signal epochs. 

    Parameters
    ---------- 
    electrode_slices: dict
        Key is an event name, values are signal epochs.
    exact_sr: float
        exact sampling rate info from the EEG amplifier
    freq_min, freq_max : int, optional
        lower and upper frequency limits for plotting (default 0, 50 Hz)
    mode: {'period', 'welch'}
        Default is period which produces periodogram. 
        change to 'welch' for alternative method of power estimation.
    name : str, optional
        title of the figure    

    Returns
    -------
    power_density : dict
        keys are event names. Under each key there are two np.arrays.
        One array stores info about frequency bins, second has power densities for all trials separately.
        

    """
    
    sns.set()
    sns.set_palette("hls")
    fig, axes = plt.subplots(1)
    fig.suptitle(name)
    print(name)
    colors = ['r','g', 'b', 'yellow', 'm', 'orange']
    color_dict =  {name: colors[i] for i, name in enumerate(electrode_slices.keys())}
    
    power_density = {}    
    
    for name, event in electrode_slices.items():

        if(mode=='welch'):
            f, Pxx_den = signal.welch(event, exact_sr, nperseg=512)
        elif mode=='period':
            f, Pxx_den = signal.periodogram(event, exact_sr)
        
#        elif mode=='fft':
#            n = len(event[0,:])
#            f = frq[range(n/2)]            
#            Pxx_den = np.fft.fft(event)/n 
#      
        
        min_idx = np.argmax(f > freq_min)
        max_idx = np.argmax(f > freq_max)
        
        g = sns.tsplot(data=Pxx_den[:,min_idx:max_idx], time = f[min_idx:max_idx],  err_style="unit_traces", condition = name, color =color_dict[name], ax = axes)
       # g.fig.suptitle()
        axes.set_yticklabels(labels = f[min_idx: max_idx], rotation = 0)

        axes.set_ylabel(mode+' Power Density')
        axes.set_xlabel('frequency')
    
        power_density[name] = (f[min_idx:max_idx], Pxx_den[:,min_idx:max_idx])
    return power_density
    

    
def PlotErp(electrode_slices, n_back):
    """Plot average and individual traces of epoched signal per condition. 

    Parameters
    ----------
    electrode_slices: dict
        Key is a condition label, contains array of signal epochs.
    n_back: int
        information about the time of event from the end of the epoch.
    """
    sns.set()
    fig, axes = plt.subplots(1)
    #event_avg = []
    for name, event in electrode_slices.items():
        #event_avg.append(np.mean(event, axis = 0))
        if('events_Erp' not in locals()):
            events_Erp = event
        else:
            events_Erp = np.dstack((events_Erp , event))   
    
    sns.tsplot(data=events_Erp, err_style="unit_traces", condition = [key for key in electrode_slices.keys()], ax = axes)
    axes.axvline(n_back, color='r', linestyle='--')       
    
def PlotSpectrogram(electrode_slices, n_back, n_forth):
    """Plot average spectrogram (time by frequency). 
       Uses n_back and n_forth to approximate the time bin where event occured.

    Parameters
    ----------
    electrode_slices: dict
        Key is a condition label, contains array of signal epochs.
    n_back: int
        information about the time of event from the end of the epoch.
    n_forth: int
        information about the time of event from the begining of the epoch.
    """
    sns.set()
    fig, axes = plt.subplots(2,2)
    i = 0
    for name, event in electrode_slices.items():
        #np.flipud, because it's uspide down
        axes[0,i].set_title(name)
        axes[0,0].set_ylabel('> 13 Hz ')
        axes[1,0].set_ylabel('0 - 13 Hz')
        
        #Sxx already after log
        f, t, Sxx, upper_bound = AverageSpectrogram(event)
        
        #Plot the upper part of the spectrum, i.e. above alpha
        g = sns.heatmap(np.flipud(Sxx)[upper_bound::,:], annot=False, robust = True, cbar = True,
                     xticklabels = t, ax = axes[0,i])
        g.set_yticklabels(labels = f[upper_bound::], rotation = 0)        

                     
        #Plot the lower part of the spectrum, i.e. below and including alpha
        g = sns.heatmap(np.flipud(Sxx)[0:upper_bound,:], annot=False, robust = True, cbar = True,
                     xticklabels = t, ax = axes[1,i])
                     
        g.set_yticklabels(labels = f[0:upper_bound], rotation = 0)
                     
        event_pos = CalcEventPosition(n_back, n_forth, len(t))
        
        axes[0,i].axvline(event_pos, color='g', linestyle='--')
        axes[1,i].axvline(event_pos, color='g', linestyle='--')                
        
        i = i +1
        
    return g
    
    
def CalcEventPosition(n_back, n_forth, n_fft_windows):
    pos = n_back / ( (n_back+n_forth) / n_fft_windows )
    #print(pos)
    return pos

def AverageSpectrogram(electrode_event):

    f, t, Sxx = signal.spectrogram(electrode_event, window = 'hamming', fs = 100,nperseg = 64, noverlap = 32, return_onesided =True, scaling = 'spectrum' )
    
    Sxx = np.mean(np.log(Sxx), axis = 0)
    
    upper_bound = np.argmax( f > 13.0 )

    f = [format(x, '.1f') for x in f]
            

    return f, t, Sxx, upper_bound



