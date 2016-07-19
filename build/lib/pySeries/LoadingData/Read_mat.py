"""
ReadData
========

Converts the data from matlab to a HDF5 data structure.
Data is stored in row-major order- where each row is a next sample.

"""

import deepdish as dd
import numpy as np
import scipy.io as sio
import glob
import os
from collections import Counter
import pandas as pd

def Load_Rest():
    """Load bands data processed by kasie.
       First dimension is electrode (59),
       Second dimension is band ('all_spec','theta','alpha', 'smr','beta12-22', 'beta15-22', 'beta22-30', 'trained', 'ratio')
       Third dimension is before/after (2)

    Returns
    -------
    DataFrame:
        tidy long format

    """

    in_path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/pasma_rest/'
    filtering_var = 'Abs_amp_OO'
    all_subjects = mat2py_read(in_path = in_path, filtering_var = filtering_var, reshape = False)
    #Select electrodes
    channels = pd.read_csv('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/channels.csv')
    ch_idx = channels[channels['Channel'].isin(['F3','F4','P3', 'P4'])].index.tolist()
    bands_dict = ['all_spec','theta','alpha', 'smr','beta12-22', 'beta15-22', 'beta22-30', 'trained', 'ratio']
    bands_dict.extend(bands_dict)
    #Add conditions info
    conditions_info = pd.read_csv('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/subjects_conditions.csv')


    tmp = []

    period =['before' for i in range(9)]
    period.extend(['after' for i in range(9)])
    for name, subject in all_subjects.items():
        #Use F as he argument to flatten to ravel around columns not rows
        bands = subject[ch_idx,:,:].mean(axis =0).flatten('F')

        df = pd.DataFrame({'band_values': bands, 'band_names': bands_dict})
        df['subject'] =name[11::]
        df['period'] = period
     #   print(name[11::])
        #TODO, make the selection nicer, why cant I access the value (the loc returns a pd.Series)
        condition =  conditions_info.loc[conditions_info['subject'] == name[11::]]['condition'].values[0]
        df['condition'] =condition
        tmp.append(df)

    return pd.concat(tmp, ignore_index = True)

def Load_Rest_Signal(filtering_var):
    in_path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/sygnal_rest/mat_format/'

    all_subjects = mat2py_read(in_path = in_path, filtering_var = filtering_var, reshape = False)

    channels = pd.read_csv('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/channels.csv')
    ch_idx = channels[channels['Channel'].isin(['F3','F4','P3', 'P4'])].index.tolist()
    print(ch_idx)


    for name, subject in all_subjects.items():
        
        selected_electrodes = {
        
        'F3' : subject[ch_idx[0], :],
        'F4' : subject[ch_idx[1], :],
        'P3' : subject[ch_idx[2], :],
        'P4' : subject[ch_idx[3], :]
        
        }

        all_subjects[name] = selected_electrodes


    return all_subjects
    
def Load_Training_Signal(filtering_var):
    in_path = '/Users/user/Desktop/Analysis projects/treningi_mat/'
    
    all_subjects = mat2py_read(in_path = in_path, filtering_var = filtering_var, reshape = True)
    
    return all_subjects



def mat2py_read(in_path = '', filtering_var = '', reshape = False):
    r"""Loads all the .mat files in the folder and converts them to HDF5 database.

    Parameters
    ----------
    in_path: str, optional
        Path to the folder with all recordings.
    filtering_var: str, optional
        When the path is a folder and you want to take only some files from it,
        provide a partial string by which the included files can be identified.

    reshape: bool, optional (default is False, which means matlab and python data structures will have the same dimensions)
        Convert to row-major order, where dimension are organized according to their length [medium, max, min]
        It is necessary to reshape the column-major data to save it in HDF5 format.

    Returns
    -------
    pyVar: ndarray
        an array of the shape of the original matlab file or (depending on rehape bool)
        a row-major array with dimensions [depth(layer),  rows,  columns]
        reshaped to organize dimensions according to their length [medium, max, min]

    Examples
    --------
    To load all neurofeedback trainings data from a single elctrode run:

    in_path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/sygnal_treningi/'
    out_path ='/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/Pickles/'
    trainings_data = mat2py_read(in_path, filtering_var = 'P4_trening_', reshape = True)
    SaveHDF(trainings_data, 'my_database_name', out_path)

    """

    full_paths = [single_recording for single_recording in glob.glob(in_path+'*') if filtering_var in single_recording]
    all_timeseries = {}
    #Enumerate instead of a simple for loop just to give information how many files are left
    for idx, name in enumerate(full_paths):
        print(name)
        #Load the matlab variable
        matVar = sio.loadmat(name)
        #extract just the variable (i.e. drop metadata) which is USUALLY undder the only key without '__' prefix (and thus always comes up last in a sorted list)
        pyVar = matVar[sorted(matVar.keys())[-1]]#.swapaxes(0,2).swapaxes(1,2)

        if(reshape):
            #Get the dimension sizes, from the largest to the smallest dimension
            shape_order = np.ndarray.argsort(np.array(pyVar.shape))[::-1]
            #Reorganize to row major order, where rows dimension is largest, then layers (depth) dimension is medium and columns dimesnion is smallest
            pyVar = pyVar.swapaxes(shape_order[0],shape_order[1]).swapaxes(shape_order[2],shape_order[1])

        #Get only the unique part of the name and drop the mat extension
        short_name = os.path.basename(name).replace(filtering_var, '').replace('.mat', '')
        #store the processed data cube in a dictionary
        all_timeseries[short_name] = pyVar
      #  print('Remaings files to load: %i'%(len(full_paths)-idx))

    #SaveHDF(all_timeseries, 'all_trainings')
    return all_timeseries

def LoadAvg(in_path, filter_list, freq_lim = None):
    """Read Data from folder with signals to be averaged.

       Apply to processed data (e.x. fft) as raw signals from different sources usually cannot be averaged.

    Parameters
    ----------
    in_path: str
        Path to folder with collections of signals.
    filter_list: list[str]
        List of strings, where each item exclusively describes a filtering term for the group to be averaged
    freq_lim: list[int](optional, default None)
        Used to load only a part of power spectrum, defined by lower and upper frequency bound
        `freq_lim[0]` is a lower frequency bound
        `freq_lim[1]` is an upper frequecny bound

    Returns
    -------
    averaged_dict: dict[ndarray]
        dicitionary containing averaged data from all the signals collection defined by `filter_list`

    """

    all_electrodes = []
    for electrode in filter_list :

        tmp = mat2py_read(in_path,electrode, reshape = True)
        for key,value in tmp.items():
            tmp[key] =(value/len(filter_list))
        all_electrodes.append(tmp)

    c= Counter()
    for d in all_electrodes:
        c.update(d)

    averaged_dict = dict(c)
    #Select only part of the power spectrum
    if(freq_lim is not None):
        freq_bounds, freq_vals = LoadFreqInfo( freq_lim[0],freq_lim[1])
        for key, value in averaged_dict.items():
            averaged_dict[key] = value[:, freq_bounds[0]:freq_bounds[1],:]

    return averaged_dict


def LoadFreqInfo(min_freq,max_freq):
    """Provide indexes for fft data, to limit it to certain range

       Indexes are not identical to frequency in fft data. More or less index = (frequency / 2) + 1

       Parameters
       ----------
       min_freq: int
           lower frequency bound to limit the fft power spectrum

       max_freq: int
           upper frequency bound to limit the fft power spectrum

       Returns
       -------
       list: [min_idx, max_idx]
           indexes in fft corresponding to frequency bounds defined by `min_freq`, `max_freq`

       Notes
       -----
       LoadFreInfo() assumes all fft's were computed with the same parameters, thus have the same freq to index relation
    """

#TODO check whats up with the overlapping subjects from tura 2 and 3, repeat when normalization amplitude methds are finally decided (divide by sum/mean, take sum/mean)
    path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/fft_treningi/'


    #Load only a single file, does not matter each one as all fft's were computed with the same parameters so contain the same frequency to index relation
    sample_file = glob.glob(path+'*')[0]

    #Get the indexes where the frequencies are in bound of those of interest. Index does not eqal frequency
    freqs = sio.loadmat(sample_file)['freqs']
    min_idx = np.where(freqs> min_freq)[0][0]
    max_idx = np.where(freqs> max_freq)[0][0]

    return [min_idx, max_idx], freqs[min_idx: max_idx]




def SaveHDF(var, name, out_path):
    """Saves dictionary of numpy arrays to HDF5 using deepdish library.

    Parameters
    ----------
    var: any
        variable to save.
    name: str
        Name to save the `var` on the disk. Usually the same as variable name.
    out_path: str
        path to save the HDF5 files.

    """

    dd.io.save(out_path + name+'.h5', var, compression=None)