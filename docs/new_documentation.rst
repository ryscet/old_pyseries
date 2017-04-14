.. _new_documentation:

=============
Documentation
=============

Pyseries uses a dictionary for storing all information collected during the experiment, we will call this dictionary "recording". Recording dictionary contains signal from eeg channels, timestamps, event markers and metadata. A screenshot of recording dictionary taken from spyder is below: 

.. image:: images/recording.png
	:scale: 30 %

An important field in this dictionary is under key "events". Here, using a pandas DataFrame, we store all information about types and timings of events like stimulus appearance or subject responses. We will refer to this DataFrame as "events".

.. image:: images/events.png
	:scale: 30 %


API
^^^
.. toctree::
   :maxdepth: 1

   Analysis
   Preprocessing
   ReadData