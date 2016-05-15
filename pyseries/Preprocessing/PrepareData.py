"""
PrepareData
===========

Filter signals using PCA decomposition of their fast-fourier transform

    #. Identify outliers in fft-transformed signal
    #. Identify clusters in fft - i.e. signals with simmilar features

"""
import sys
sys.path.insert(0, '/Users/ryszardcetnarski/Desktop/pySeries')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import pandas as pd
#import LoadingData.ReadData as rd
import obspy.signal.filter as filters
import pickle
from scipy import signal

def Run():
    in_path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/fft_treningi/'
    filter_list = ['P3_trening_','P4_trening_', 'F3_trening_', 'F4_trening_']
    #load averaged fft from trainings
    signals_dict = rd.LoadAvg(in_path, filter_list, freq_lim = [-1,40])
    #label the outliers
    labels = FilterIndividualAndGroup(signals_dict, plot_on = False, log = True)
    #Filter bad blocks and save bands
    fft_db = Prepare_FFT(labels)
    #Plot'em

    #MarkOutliers(labels[labels['mask'] ==1])

    return labels#, signals_colum_major


#def Load_rs_envelopes():
#    with open('/Users/user/Desktop/Data/sygnal_rest/processed/rest_envelopes.pickle', 'rb') as handle:
#        b = pickle.load(handle)
#    return b


def PrepareRestingSignal():
    """Convert raw signal into power spectrum (over time) and envelopes of specified frequencies.
       Frequencies are defined in a dictionary of bands, {name : (lower f, upper f)}.
       Particualr parameters of signal (fs) and fft parameters need to be adjusted downstream.

    """
    all_sigs = rd.Load_Rest_Signal('_1_OO_camil.mat')
    all_subjects_freqs = {}
    all_subjects_envelopes = {}


    bands = {'all_spec':(1,30), 'alpha':(8,11), 'beta1': (12.5,22), 'beta2':(15,22), 'beta3': (23,30)}

    for subject_name, all_electrodes in all_sigs.items():
        print(subject_name)

        freqsByTime = {}
        envelopes = {}
        for electrode_name, recording in all_electrodes.items():

            freqsByTime[electrode_name] = ExtractFrequencyTime(recording, bands)
            envelopes[electrode_name] = ExtractBandEnvelope(recording, bands)

        all_subjects_freqs[subject_name] = freqsByTime
        all_subjects_envelopes[subject_name] = envelopes

#
    with open('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/sygnal_rest/processed/rest_specgrams.pickle', 'wb') as handle:
                pickle.dump(all_subjects_freqs, handle)

    with open('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/sygnal_rest/processed/rest_envelopes.pickle', 'wb') as handle:
                pickle.dump(all_subjects_envelopes, handle)
#
#    return all_subjects_freqs, all_subjects_envelopes




def ExtractBandEnvelope(data, bands):
    """Computes the envelopes of specified bands.
    """
    bands_results = {'all_spec':[], 'alpha':[], 'beta1':[], 'beta2':[], 'beta3':[]}

    for name, band_range in bands.items():

        #f, axes = plt.subplots(4, sharex=True, sharey=True)
        filt = FilterData(data, band_range[0], band_range[1])
        envelope = filters.envelope(filt)
        bands_results[name]= envelope

    return bands_results


def ExtractFrequencyTime(data, bands):
    """Computes the spectrogram of data, and the takes the sum of frequencies inside the band range.
    """

    bands_results = {'all_spec':[], 'alpha':[], 'beta1':[], 'beta2':[], 'beta3':[]}

    f, t, Sxx = signal.spectrogram(data,window = 'hamming', fs = 500,nperseg = 256, noverlap = 128, return_onesided =True, scaling = 'spectrum' )

    # plt.pcolormesh(t, f, np.log(Sxx))



    for name, band_range in bands.items():
        min_idx = np.where(f > band_range[0] - 0.3)[0][0]
        max_idx = np.where(f> band_range[1])[0][0]
        print(name + ' min  %f max %f' %(f[min_idx], f[max_idx]))

        bands_results[name] = np.sum(Sxx[min_idx : max_idx +1, :], axis =0)

    return bands_results



def convert_to_col_major(minFreq = -1,maxFreq =50):
    #I.e. forege about division on subjects and sessions, just block after block (each row is a block, each column a freqw)
    in_path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/fft_treningi/'
    filter_list = ['P3_trening_','P4_trening_', 'F3_trening_', 'F4_trening_']

    signals_dict = rd.LoadAvg(in_path, filter_list, freq_lim = [minFreq,maxFreq])

    for name, signals_array in signals_dict.items():

        nans = np.isnan(signals_array)
        subject_signals = signals_array[~nans.any(1).all(1)]

        subject_signals = np.log(subject_signals)

        flattened = subject_signals.transpose(0,2,1).reshape(-1,subject_signals.shape[1])

        #Store data from all subject in one structure

        if('all_signals' not in locals()):
            all_signals = np.array(flattened)

        else:
            all_signals = np.vstack((all_signals, flattened))

    return all_signals



def Prepare_treningi(labels):
    """
       a.k.a. Load_Treningi
       Combines pca filtering results with conditions info.
       Reloads fft because pca filtring was done on 15-40 range, and the result should have all.

      Parameters
      ----------

      labels: DataFrame
          Describe each block as in/outlier after PCA filtering

      Returns
      -------
      combined: DataFrame
          wide format df of bands with outliers removed
    """
    #cannot load labels because they can come out in different order each time spyder is run. Why??!??!
    conditions_info = pd.read_csv('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/subjects_conditions.csv')
    signals_colum_major = convert_to_col_major()


    combined = pd.concat([labels, pd.DataFrame(signals_colum_major)], axis =1)
    #NOW DROP THE BAD BLOCKS
    combined = combined[combined['mask']==0]


    combined['condition'] = ''
    for idx, row in conditions_info.iterrows():
        combined.loc[combined['subject'] == row['subject'], 'condition'] = row['condition']

    bands =  {'theta':(4,8),'alpha': (8,12), 'smr': (12,15), 'beta1':(15,22), 'beta2':(22,30), 'trained':(12,22)}

    for band_name, band_range in bands.items():
        print('hello')
        print(band_name)

        freq_idx, freq_vals = rd.LoadFreqInfo(band_range[0],band_range[1])

        indexes = [i for i in range(freq_idx[0], freq_idx[1])]

        band_columns =combined[indexes]
        print(freq_vals)
        print(indexes)
        combined[band_name] =  band_columns.mean(axis = 1)

    cols =[col for col in combined.columns if type(col) == int]
    combined = combined.drop(cols,axis=1)

    combined.to_pickle('/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/bands.pickle')
    return combined


def MarkOutliers(bad_ones):
    r"""Plots the results of two-step outlier pca classification

        Parameters
        ----------
        bad_ones: DataFrame
            labels of outlier blocks created in previous steps (`FilterIndividualAndGroup()`)
    """
    in_path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/fft_treningi/'

    minFreq, maxFreq = 0,30 #Just for plotting
    filter_list = ['P3_trening_','P4_trening_', 'F3_trening_', 'F4_trening_']

    signals_dict = rd.LoadAvg(in_path, filter_list, freq_lim = [minFreq,maxFreq])

    freq_idx, freq_vals = rd.LoadFreqInfo(minFreq,maxFreq)

    for name, fft in signals_dict.items():
        fig= plt.figure()
        fig.suptitle(name, fontweight = 'bold')
        ax = fig.add_subplot(111)

        #print('plotting %s' %name)
        for session in range(0, fft.shape[0]):

            for block in range(0,fft.shape[2]):

                single_fft = fft[session,:,block][~np.isnan(fft[session,:,block])]
                if(np.count_nonzero(~np.isnan(single_fft)) > 1):
                    color = 'blue'
                    if(bad_ones['alltogether'].str.contains(name +str(session)+str(block)).any()):
                        color = 'red'
                    ax.plot(freq_vals, np.log(single_fft),  color = color, alpha = 0.2)




def FilterIndividualAndGroup(signals_dict, plot_on =False, log = False):
    r"""Iterates over dict of data samples performing two step PCA filtering

        First step is to remove outliers identivfied in individually grouped data
        Second step identifies two clusters in the PCA decomposition of all agregated data samples

    Parameters
    ----------
    sigs_dict: dict
        containing ndarrays of signals.

    log: bool (optional, default False)
        whether to transform data by log. Useful when dealing with power spectrum of EEG data

    plot_on: bool (optional, default True)
        if True then a plot of PCA results will be produced.

    Returns
    -------
    all_labels, all_signals: pandas.DataFrame, ndarray
        all_labels marks each block as outliers (all_labels['mask'] == 1) or inlier (all_labels['mask'] == 0)
        all_signals is a column-major collection of all signals from orignal `sigs_dict`
        Each row in all signals corresponds by index to a description in all_labels


    """
    #loads a  dict of subjects, each subject is a list of arrays, each array are blocks*session ( Session, signalIdx, block)


    #IMPORTANT COMMENT - run this to create input to this function - signals_dict
    #Slightly less importat - nan blocks are filtered but their indexes are not saved, -SOLVE
    #signals_dict, freqs = Load_FFT(15, 40)
    #Will store data frames, prepared to store in/outlier info for all blocks/session/subjects
    all_labels = []
    #Will store all in/outliers markers returned by individual pca filtering
    individual_mask_arrays = []
    for name, signals_array in signals_dict.items():
        #Filter out layers of nans, i.e. missing sessions
        #Mark, not useful for general purpose but not harmful
        #Eliminate the whole missing sessions
        nans = np.isnan(signals_array)
        subject_signals = signals_array[~nans.any(1).all(1)]
        #Log transform data before pca, usefull if the the data is a power spectrum
        if(log == True):
            subject_signals = np.log(subject_signals)

        #Flatten from 3D to 2D, now each column is a block and each row is a sample, sessions were the flattend dimension
        flattened = subject_signals.transpose(0,2,1).reshape(-1,subject_signals.shape[1])
        #Filter individual blocks of nans, happened when not a full session was permittef
        flattened_no_nans = flattened[~np.isnan(flattened).all(axis=1)]
        nan_idx = np.where(np.isnan(flattened).all(axis=1))[0]
        #Pca takes array without nan blocks as input. To mark all the blocks with in/outlier information all blocks including nans have to be annotated.
        #To restore information about nan blocks from pca output we store their index andput it back at original index as 1 - same as outlier but different, maybe not a good solution
        #Then we add info at this index to pca output. The index has to be original index - position in the nan index list, because when inserting using numpy insert every next insert lands one index ahead of its original value
        nan_idx = [nan_idx[i] - i for i in range(nan_idx.size)]

        #Data is stored in 3D cubes, we need to flatten two dimensions to get the total number of blocks
        #This is needed because we will mark each block as an inlier or outlier, so this flat list will be used for sucha annotation - labels dataframe is exactly that list
        names = [name for i in range(subject_signals.shape[0] * subject_signals.shape[2])]
        sessions = [session for session in range(subject_signals.shape[0]) for block in range(subject_signals.shape[2])]
        blocks = [block for block in range(subject_signals.shape[0]) for block  in range(subject_signals.shape[2])]
        #Prepare data frame for annotating which block from which session from which subject is an outlier
        labels = pd.DataFrame( {'subject': names, 'session': sessions, 'block':blocks})
        all_labels.append(labels)

        #Perform pca based on all blocks from subject to filter out outliers
        subject_pca_results = PcaFilter(np.array(flattened_no_nans), 'outlier', False, name)
        #print('outlier filtering: %s, n outliers: %i'%(name, sum(subject_pca_results )))

        #Add information about nan blocks
        subject_pca_results = np.insert(subject_pca_results, nan_idx, 1)

        #Store data from all subject in one structure
        individual_mask_arrays.extend(subject_pca_results)
        if('all_signals' not in locals()):
            all_signals = np.array(flattened)

        else:
            all_signals = np.vstack((all_signals, flattened))

    individual_mask_arrays = np.array(individual_mask_arrays)
    print('\nN outliers inside individual: %i, %f %%' %(sum(individual_mask_arrays),(sum(individual_mask_arrays)/len(individual_mask_arrays))))

    #PCA filtering on data from all subjects combined will be perforemd after deleting the individual outliers
    #This will improve accuracy of filtering
    filtered = all_signals[np.where(individual_mask_arrays == 0)[0],:]


    grouped_mask_array = PcaFilter(filtered,'cluster', plot_on, 'all subjects')
    print('\nN outliers across group: %i, %f %%' %(sum(grouped_mask_array),(sum(grouped_mask_array)/len((grouped_mask_array)))))

    #Combine the score from individual and group filtering
    all_labels = pd.concat(all_labels, ignore_index = True)
    individual_mask_arrays[np.where(individual_mask_arrays == 0)[0]] = individual_mask_arrays[np.where(individual_mask_arrays== 0)[0]] + grouped_mask_array
    all_labels['mask']= individual_mask_arrays
    #Var used by markBadOnes Function, dirty hack actually
    all_labels['alltogether']= all_labels['subject'] + all_labels['session'].map(str)+ all_labels['block'].map(str)

    return all_labels#, all_signals





def PcaFilter(sigs_array, method, plot_on=False, name='' ):
    r"""Filter out a subset of signals.

    Parameters
    ----------
    sigs_array: list[ndarray]
        ndarray of containing signals. Has to be in row-major order.

    method, str: {'outlier', 'cluster'}
        If 'outlier' then exclude signals based on malahanobis distance of theri PCA transform.
        If 'cluster' exclude the smaller group resulting from k-means clustering.

    plot_on: bool
        if True then a plot of PCA results will be produced.

    name: str (optional)
        use to label the plots when more then on are produced.

    Returns
    -------
    mask_array: ndarray
        array of 0's and 1's of the lenght of `sigs_array`. o's are inlier and 1's are outlier signals.

    Notes
    -----
    Cluster method for PCA filtering **assumes** that members of the smaller cluster are outliers.
    Cluster method falso  assumes that there will be only two clusters: inliers and outliers.

    """
    #Pca decomposition into first two components
    sklearn_pca = sklearnPCA(n_components=2)
    pcs = sklearn_pca.fit_transform(sigs_array)


    if(method == 'outlier'):
        #This index corresponds to the original index on the array of time series
        #last argument is the threshold of how many standard deviations away a point is considered an outlier
        outlier_idx = MD_removeOutliers(pcs[:,0], pcs[:,1], 2)
        #this will be used for a boolean array for filtering.
        mask_array = np.zeros(len(pcs))
        if(len(outlier_idx) >0 ):
            mask_array[outlier_idx] = 1


    if (method == 'cluster'):
        mask_array = Cluster(pcs, plot_on)

    if(plot_on):
        colors = ['r', 'b']
        fig = plt.figure()
        fig.suptitle('PCA decomposition')
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        #Plot PCA scores and mark outliers
        ax2.scatter(pcs[:,0], pcs[:,1], c = mask_array, cmap = 'jet', s = 60, marker = 'o')
        #Print variance ratio
        ax2.annotate(sklearn_pca.explained_variance_ratio_,xy= (1,1), xycoords='axes fraction', horizontalalignment='right', verticalalignment='top')
        #Plot original signals and mark PCA indentified outliers
        for idx,row in enumerate(sigs_array):
            ax1.plot(row, color =colors[int(mask_array[idx])], alpha = 0.1)

        ax2.set_ylabel('1st principal component')
        ax2.set_xlabel('2nd principal component')

        fig.tight_layout()

    return mask_array

def Cluster(X, plot_on):
    r"""Cluster a 2D array into two sets.

    Uses silhouette score to measure clustering quality.

    Parameters
    ----------
    X: ndarray
        2D array of x and y coordinates, each row is an x,y pair.
    plot_on: bool
        If true show the clustering results and silhouette's scores on a plot.

    Returns
    -------
    mask_array: ndarray
        array of 0's and 1's of the lenght of `sigs_array`. o's are inlier and 1's are outlier signals.

    Notes
    -----

    Silhouette scores are only displayed on the plot, not actually used to move points between clusters.


    """
    n_clusters = 2

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=2, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("\nFor n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)


    if(plot_on):

        # Create a subplot with 1 row and 2 columns
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

       # ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhoutte score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=200)

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        ax2.set_xlabel("1st principal component")
        ax2.set_ylabel("2nd principal component")

      #  ax1.set_title(("Silhouette analysis for KMeans clustering on pca scores "
      #                "with n_clusters = %d" % n_clusters))
        fig.tight_layout()

        #Clusters will switch labels depending on random initialization
        #If there are mostly ones - i.e. inliers are marked as 1, reverse
    if(np.count_nonzero(cluster_labels.astype(int)) > len(cluster_labels.astype(int))/2 ):
        #reverse 0's and 1's
        mask_array = np.zeros(len(cluster_labels.astype(int))) + 1 - cluster_labels.astype(int)
    else:
        mask_array = cluster_labels.astype(int)

    return mask_array





def averageSignals(sig_list):
    r"""Takes element-wise average of ndarrays.

    `sig_list` must contain arrays of the same shape.

    Parameters
    ----------
    sig_list: list
        List of ndarrays containing collection of signals

    Returns
    -------
    avg: ndarray
        an array where each element is an average from the corresponding index in the input arrays (`sig_list`)

    Example
    -------

    >>> averageSignals([[2,2,4], [4,4,4]])
    >>> [3,3,4]

    """


#Empty array in the shape of the first element
    _sum = np.zeros(sig_list[0].shape)
#Sum and divide by n elements
    for electrode in sig_list:
        _sum = _sum + electrode
    avg = _sum / len(sig_list)

    return avg


def MD_removeOutliers(x, y, n_std):
    r"""Removes outliers from a 2D array based on their Mahalanobis distance.

    `sig_list` must contain arrays of the same shape.

    Parameters
    ----------
    x: ndarray
        1D array or list of x-coordinates
    y: ndarray
        1D array or list of y-coordinates
    n_std: float
        Threshold deciding how many standard deviations away from the mean a point is considered an outlier
        if x > avg + std * n_std

    Returns
    -------
    outliers: ndarray
        indexes of outliers found in [x,y]


    """

        #Std - how many standard deviations avay from the mean to exclude
    MD = MahalanobisDist(x, y)
    threshold = np.mean(MD)+np.std(MD)*n_std # adjust 1.5 accordingly
    nx, ny, outliers = [], [], []
    for i in range(len(MD)):
        if MD[i] <= threshold:
            nx.append(x[i])
            ny.append(y[i])
        else:
            outliers.append(i) # position of removed pair
    return np.array(outliers)



def MahalanobisDist(x, y):
    r"""Calculates mhalanobis distance in 2D for each point from input arrays.

    Parameters
    ----------
    x: ndarray
        1D array or list of x-coordinates
    y: ndarray
        1D array or list of y-coordinates


    Returns
    -------
    md: ndarray
        malahanobis distance for each point described by a pair `x` an `y`

    """

    covariance_xy = np.cov(x,y, rowvar=0)
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    xy_mean = np.mean(x),np.mean(y)
    x_diff = np.array([x_i - xy_mean[0] for x_i in x])
    y_diff = np.array([y_i - xy_mean[1] for y_i in y])
    diff_xy = np.transpose([x_diff, y_diff])

    md = []
    for i in range(len(diff_xy)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i])))
    return md


def FilterData(channel, _freqmin, _freqmax):
    b_pass = filters.bandpass(channel, freqmin = _freqmin, freqmax = _freqmax, df = 250)
   # b_stop =filters.bandstop(b_pass, freqmin = 49 ,freqmax = 51, df = 500)
    return b_pass
