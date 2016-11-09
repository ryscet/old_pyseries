# -*- coding: utf-8 -*-
"""#Michal wprowadza zmiany
## test test test
#Dodajemy bullshit
Read_edf
========
Reduced version for loading files. Reads a .edf file which contains sensor informations.
Does not include time information for events synchronization.
Comment to test meld
"""

import pyedflib 


def Read_EDF(path):
    """ Reads the .edf file and converts it into a dict of numpy arrays.
       
        Parameters
        ----------
        path:str
            path to .edf
  
        Returns
        -------
        signal_dict: dict(np.array)
            Keys are channel/sensor names
            Values are channel/sensor time-series
    """
    # creates the pyedflib object
    # The pyedflib object has many functions which provide info about signal, sample rate, filters etc. 
    # for example methods f.getStartdatetime() and f.getSignalHeader(channel index)

    f = pyedflib.EdfReader(path)
    # Gets the names of EEG channels and other types of signals 
    signal_labels = f.getSignalLabels()
        

    signal_dict = {}
    
    # idx is used to retrieve the signal at the same index as the name of he signal
    for idx, name in enumerate(signal_labels):
        signal_dict[name.decode("utf-8")] = f.readSignal(idx)
        
    # Not closing will make it impossible to open new files
    f._close()
    
    return signal_dict
#

        

    

  
