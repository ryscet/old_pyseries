pyseries
--------
pyseries is a python package for EEG signal analysis. It provides API for data i/o, analysis and visualization. GUI is on the way, look at the minimal branch of the project.

__Documentation__

* The documentation is hosted on http://pyseries.readthedocs.io/en/master/


* Note you can change between documentation for different branches (master & minimal) using the button in lower right corner on read the docs:


<img src="https://github.com/ryscet/pyseries/blob/master/docs/images/rtd.png">


Functions overview
------------------

* __Exploaratory analysis__: You can load edf data, annotate signal with markers and compare the power spectra in different conditions with a few lines of code. 

<img style="float: left;" src="https://github.com/ryscet/pyseries/blob/master/docs/images/explore.png">


* __Outliers filtering__: use PCA, clustering and silhouette analysis to find outliers in epoched data.


<img style="float: left;" src="https://github.com/ryscet/pyseries/blob/master/docs/images/pca.png">


* __Stats__: there is an ANOVA function because there was nothing to our satisfaction on scipy or statsmodels.


<img style="float: left;" src="https://github.com/ryscet/pyseries/blob/master/docs/images/anova.png">

