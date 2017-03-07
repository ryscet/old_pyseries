"""Event related potentials - averages time locked to an event"""


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

def plot_erp(epochs_dict, electrode):
    #colors = {}
    #colors['wb_Diode'] = 'b'
    #colors['bw_Diode'] = 'b'
    #colors['bw_Port'] = 'g'
    #colors['wb_Port'] = 'g'
    fig, axes = plt.subplots()
	
    sample_duration = 1000 / 497.975590198584
    time_line = np.arange(-100, 100, 1) * sample_duration
	#fig2, axes2 = plt.subplots()
    for key, value in  epochs_dict[electrode].items():
        #g = sns.tsplot(data=value, time =  time_line,  color = colors[key], condition = key, ax = axes)
        g = sns.tsplot(data=value, time =  time_line,  condition = key, ax = axes)
        break
		#g2 = sns.tsplot(data=value, time = np.arange(0, len(value[0,:]), 1),  color = colors[key], condition = key, ax = axes2, err_style = 'unit_traces')
	#axes.legend()
    axes.axvline(0, linestyle = '--', c = 'black', label = 'switch')

    axes.set_xlabel('Millieconds from event marker')
    axes.legend()