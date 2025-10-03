import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

#reading data

data = pd.read_csv("data/housing.csv")

## Quick look at the data

# print(data.head()) #
# print(data.columns) # prints names of columns
# print(data['ocean_proximity']) # returns list of values under col "ocean proximity"

# how to view data? -> use histograms
# adding a .hist() to end of "data" will give histogram (in plots field)

data['ocean_proximity'].hist()
# data.hist() #makes everything into one plot img

# how do we make the 'ocean proximity' data numerical?
# -> use SciKit-learn


# Removing Missing Data
data = data.dropna().reset_index(drop=True) #missing data saved in var "data"  

#create instance of function OneHotEncoder
enc = OneHotEncoder(sparse_output=False)
enc.fit(data[['ocean_proximity']])

# next we need to delete 'ocean proximity' col in dataset and create new columns


encoded_data = enc.transform(data[['ocean_proximity']])

category_names = enc.get_feature_names_out()

encoded_data_df = pd.DataFrame(encoded_data, columns=category_names)
data = pd.concat([data, encoded_data_df], axis=1)

data = data.drop(columns = 'ocean_proximity')

data.to_csv("revised_data.csv")



from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
"""Data Splitting"""
#this is a very basic method of data splitting
# X_train, X_test, y_train, y_test = train_test_split(X,y
#                                                          test_size = 0.2,
#                                                          random_state = 42)

#the use of stratified sampling is strongly recommended
data["income_categories"] = pd.cut(data["median_income"],
                                   bins=[0, 2, 4, 6, np.inf],
                                   labels=[1, 2, 3, 4])
my_splitter = StratifiedShuffleSplit(n_splits =1,
                                     test_size = 0.2, # % of data taken
                                     random_state = 42) # minecraft seed
for train_index, test_index in my_splitter.split(data,data["income_categories"]):
    strat_data_train = data.loc[train_index].reset_index(drop=True)
    strat_data_test = data.loc[test_index].reset_index(drop=True)
strat_data_train = strat_data_train.drop(columns=["income_categories"],axis=1)
strat_data_test = strat_data_test.drop(columns=["income_categories"],axis=1)



## data cleaning and preprocessing

# scaling needs to be done on trained data
# -min max 
"""Scaling"""

from sklearn.preprocessing import StandardScaler

my_scaler = StandardScaler()
my_scaler.fit(strat_data_train.iloc[:,0:-5]) 

# iloc = index locate - first ":" selects all rows, 
#                   "0:-5" selects columns starting from end 
#                   (negative indices start from end columns)

scaled_data_train = my_scaler.transform(strat_data_train.iloc[:,0:-5])
scaled_data_train_df = pd.DataFrame(scaled_data_train, columns=strat_data_train.columns[:,0:-5])

strat_data_train = scaled_data_train.df.join(strat_data_train.iloc[:,-5:])


## Variable Selection
# Identify y and X. Remember the goal is to find f(.) such that y=f(X)
y_train = strat_data_train['median_house_value']
X_train = strat_data_train.drop(columns=['median_house_values'])
y_test = strat_data_test['median_house_value']
X_test = strat_data_test.drop(column=['median_house_value'])


print(data.shape)
print(strat_data_train.shape)
print(strat_data_test.shape)


# looking at the colinearity of variables
corr_matrix = strat_data_train.corr()


# plot correlation matrix

corr_matrix = data.corr() # calculates correlation matrix for the entire dataset
                            #also uses entire data, including test data
                            # use strat_data_train to solve 

import matplotlib.pyplot as plt #import matlab typeshit plotting
import seaborn as sns #seaborn is an advanced plotting library

sns.heatmap(np.abs(corr_matrix)) # if you add np.abs() to the corr_matrix it will return 0-1 instead of -1 to 1

# Mask correlations above a threshold 
masked_corr_matrix = np.abs(corr_matrix) < 0.8
sns.heatmap(masked_corr_matrix)

