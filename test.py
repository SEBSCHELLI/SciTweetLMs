import os
import pandas as pd

if 'SciTweetLMs' not in os.getcwd():
    os.chdir('home/schellsn/python/SciTweetLMs')

data = pd.read_csv('annotations.tsv', sep='\t')