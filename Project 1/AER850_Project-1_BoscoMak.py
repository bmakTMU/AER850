## import libraries

# 2.1
import pandas as pd
import numpy as np

# 2.2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

""" 2.1 Data processing - read data from .csv file """

data = pd.read_csv("data/Project 1 Data.csv")

## sanity check

# print(data.head()) 
# print(data.columns)
# print(data['X'])
# # data['X'].hist()
# data['Step'].hist()


""" 2.2 Data Visualization - """

## split data based on step value

data['coordinates'] = pd.cut(data['Step'],
                             bins = [0, 8, 9, 10, np.inf], # 4 groups - 1-7, 8, 9, 10-13
                             labels =[1, 2, 3, 4], # bin labels
                             right=False) 
splitter = StratifiedShuffleSplit(n_splits=1,
                                  test_size = 0.2, # % of data taken
                                  random_state = 21) # seed
for train_index, test_index in splitter.split(data,data['Step']): # split loop
    strat_data_train = data.loc[train_index].reset_index(drop=True) # splits training data
    strat_data_test = data.loc[test_index].reset_index(drop=True) # splits testing data
#drop extra col

strat_data_train = strat_data_train.drop(columns=["coordinates"],axis=1)
strat_data_test = strat_data_test.drop(columns=["coordinates"],axis=1)

# ## sanity check

# data.hist()
# strat_data_train.hist()
# strat_data_test.hist()

""" Data Scaling """

scaler = StandardScaler()
scaler.fit(strat_data_train.iloc[:, 0:-5])
scaled_data_train = scaler.transform(strat_data_train.iloc[:, 0:-5])
scaled_data_train_df = pd.DataFrame(scaled_data_train, columns = strat_data_train,columns[:, 0:-5])
strat_data_train = scaled_data_train.df































