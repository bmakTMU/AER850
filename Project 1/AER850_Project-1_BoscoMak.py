## import libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt # 2.2
from sklearn.model_selection import train_test_split, StratisfiedShuffleSplit

""" 2.1 Data processing - read data from .csv file """

data = pd.read_csv("data/Project 1 Data.csv")

## sanity check

# print(data.head()) 
# print(data.columns)
# print(data['X'])
# data['X'].hist()
# data.hist()


""" 2.2 Data Visualization - """

## split data 

data[""]