PCA filtering
-------------

Filter signals using PCA decomposition (:func:`sklearn.decomposition.PCA`) of their fast-fourier transform

    #. Identify outliers in fft-transformed signal with :func:`PrepareData.MD_removeOutliers`
    #. Identify clusters in ft-transformed signal with  :func:`PrepareData.Cluster`



.. code:: python

    import LoadingData.ReadData as rd
    import Preprocessing.PrepareData as pd

.. code:: python

    in_path = '/Users/ryszardcetnarski/Desktop/Nencki/Badanie_NFB/Dane/fft_treningi/'
    filter_list = ['P3_trening_','P4_trening_', 'F3_trening_', 'F4_trening_']
    #load averaged fft from trainings
    signals_dict = rd.LoadAvg(in_path, filter_list, freq_lim = [15,40])
    #label the outliers
    labels, signals_colum_major = pd.FilterIndividualAndGroup(signals_dict, plot_on = True, log = True)
    #Plot'em
    pd.MarkOutliers(labels[labels['mask'] ==1], rd.LoadAvg(in_path, filter_list, freq_lim = [0,50]))



.. parsed-literal::
    
    N outliers inside individual: 338, 0.039348 %
    
    For n_clusters = 2 The average silhouette_score is : 0.735854433865
    
    N outliers across group: 539, 0.065317 %


.. image:: output_1_2.png



.. image:: output_1_3.png



.. image:: output_1_4.png



.. image:: output_1_5.png



.. image:: output_1_6.png



.. image:: output_1_7.png



.. image:: output_1_8.png



.. image:: output_1_9.png



.. image:: output_1_10.png



.. image:: output_1_11.png



.. image:: output_1_12.png



.. image:: output_1_13.png



.. image:: output_1_14.png



.. image:: output_1_15.png



.. image:: output_1_16.png



.. image:: output_1_17.png



.. image:: output_1_18.png



.. image:: output_1_19.png



.. image:: output_1_20.png



.. image:: output_1_21.png



.. image:: output_1_22.png



.. image:: output_1_23.png



.. image:: output_1_24.png



.. image:: output_1_25.png



.. image:: output_1_26.png



.. image:: output_1_27.png



.. image:: output_1_28.png



.. image:: output_1_29.png



.. image:: output_1_30.png



.. image:: output_1_31.png



.. image:: output_1_32.png



.. image:: output_1_33.png



.. image:: output_1_34.png



.. image:: output_1_35.png



.. image:: output_1_36.png



.. image:: output_1_37.png



.. image:: output_1_38.png



.. image:: output_1_39.png



.. image:: output_1_40.png



.. image:: output_1_41.png



.. image:: output_1_42.png



.. image:: output_1_43.png



.. image:: output_1_44.png



.. image:: output_1_45.png



.. image:: output_1_46.png



.. image:: output_1_47.png



.. image:: output_1_48.png



.. image:: output_1_49.png



.. image:: output_1_50.png



.. image:: output_1_51.png



.. image:: output_1_52.png



.. image:: output_1_53.png



.. image:: output_1_54.png


