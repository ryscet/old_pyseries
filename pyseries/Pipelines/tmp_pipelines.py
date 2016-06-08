# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:46:31 2016

@author: user
"""


path = '/Users/user/Desktop/Piloty EEG Nencki /Ola_Intern/31_05/'



        


def Run():
    n_back =  500
    n_forth = 12
    
    electrode_slices = Make_Slices_for_Channel(['EEG O2'], n_back, n_forth )

    PlotErp(electrode_slices['EEG O2'], n_back)
    
    PlotSpectrogram(electrode_slices['EEG O2'], n_back, n_forth)
    
    PlotPowerSpectrum(electrode_slices['EEG O2'])



      




    