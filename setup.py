import os
from setuptools import setup
import setuptools

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pyseries",
    version = "1.0.19",
    author = "Ryszard Cetnarski",
    author_email = "cetnarski.ryszard@gmail.com",
    long_description="Developed for analysis of EEG recordings. Targeted for neuro and cognitive science academics looking for a quick start into EEG data analysis with python.", 
    description = ("pySeries is a package for statistical analysis of time-series data."),
    license = "MIT",
    keywords = "time, series, signal, analysis, statistics",
    url = "http://pyseries.readthedocs.io/en/latest/",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research"
    ],
    packages=setuptools.find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn', 'deepdish', 'obspy', 'tabulate'],

)