"""Event related potentials - averages time locked to an event"""


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

def plot_erp(epochs_dict):
	for key, value in  epochs_dict.items():
		plt.plot(value.mean(axis = 0))