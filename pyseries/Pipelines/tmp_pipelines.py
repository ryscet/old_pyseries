# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:46:31 2016

@author: user
"""


path = '/Users/user/Desktop/Piloty EEG Nencki /Ola_Intern/31_05/'

def AverageSpectrogram(electrode_event):

    f, t, Sxx = signal.spectrogram(electrode_event, window = 'hamming', fs = 100,nperseg = 64, noverlap = 32, return_onesided =True, scaling = 'spectrum' )
    
    Sxx = np.mean(np.log(Sxx), axis = 0)
    
    upper_bound = np.argmax( f > 13.0 )

    f = [format(x, '.1f') for x in f]
            

    return f, t, Sxx, upper_bound


def Cut_Slices_From_Signal(signal, events, n_samples_back, n_samples_forth):
#Iterates through all events and finds the index of the closest EEG sample in the signal. 
#Takes a window spanning back and forth around the closest sample. 
    slices = np.zeros((len(events.index), n_samples_back + n_samples_forth))    
    for idx, timestamp in enumerate(events.index):
        #Finding first index greater than event timestamp produces more error than finding the closest data timestamp regrdless of whether it is larger or smaller than event.
        event_index = Find_Closest_Sample(signal,timestamp)
        print('event index: %i'%event_index)
        #TODO check how inclusion/exclusion of lower and iupper bounds might affect the seslected slice size
        slices[idx,:] = signal.iloc[event_index - n_samples_back : event_index + n_samples_forth]
    
    return slices


def Make_Slices_Groups(data, events, n_samples_back, n_samples_forth):
    """Loads signal and events and creates a list of np.arrays per event type
    
        Parameters
        ----------
        data: np.array
            whole EEG signal
        events: DataFrame
            Timestamps and description of events
        n_samples_back, n_samples_forth: (int, int)
            window size around time of event
        
        Returns
        -------
        grouped slices: list(np.arra)
            list of arrays, one per event type
        
    """

    grouped_slices = {}   
    
    for name, events_single_type in events.groupby('code'):
        grouped_slices[str(name)] = Cut_Slices_From_Signal(data, events_single_type, n_samples_back, n_samples_forth)    
    return grouped_slices


def mark_events():
    db = Combine_EDF_XML(path)
    fig, axes = plt.subplots(1,1)
    axes.plot(db["timestamp"], db["EEG O1"])
    for idx, row in db["events"].iterrows():
        axes.axvline(idx, color='r', linestyle='--')
        #print(row.index)
        


def Run():
    n_back =  500
    n_forth = 12
    
    electrode_slices = Make_Slices_for_Channel(['EEG O2'], n_back, n_forth )

    PlotErp(electrode_slices['EEG O2'], n_back)
    
    PlotSpectrogram(electrode_slices['EEG O2'], n_back, n_forth)
    
    PlotPowerSpectrum(electrode_slices['EEG O2'])

def Make_Slices_for_Channel(ch_names,n_samples_back, n_samples_forth):
    channels = Combine_EDF_XML(path)
    events = channels["events"]
    
    electrode_slices = {}
    
    for name in ch_names:
        ts =pd.Series(np.array(channels[name], dtype = 'float32'), index = channels['timestamp'] )
        electrode_slices[name] = Make_Slices_Groups(ts, events, n_samples_back, n_samples_forth)
    
    return electrode_slices

#electrode_slices = Make_Slices_for_Channel(['EEG Fpz-Cz'])


def PlotErp(electrode_slices, n_back):
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

def CalcEventPosition(n_back, n_forth, n_fft_windows):
    pos = n_back / ( (n_back+n_forth) / n_fft_windows )
    #print(pos)
    return pos

def PlotSpectrogram(electrode_slices, n_back, n_forth):
    
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

def PlotPowerSpectrum(electrode_slices):
    sns.set()
    fig, axes = plt.subplots(1)
    for name, event in electrode_slices.items():
    
        f, Pxx_den = signal.welch(event, 100, nperseg=32)
        #avg_Pxx = np.mean(Pxx_den, axis = 0)        
        
        if('condtions_Pxx' not in locals()):
            condtions_Pxx = Pxx_den
        else:
            condtions_Pxx = np.dstack((condtions_Pxx , Pxx_den))       
    
    sns.tsplot(data=np.log(condtions_Pxx), time = f,  err_style="unit_traces", condition = [key for key in electrode_slices.keys()], ax = axes)
                     
    axes.set_yticklabels(labels = f, rotation = 0)
    
    axes.set_ylabel('Welch Power Density')
    axes.set_xlabel('frequency')



    