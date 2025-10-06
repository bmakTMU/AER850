""" AER850 Project 1 - Bosco Mak 501104446 """

""" 2.1 Data processing - read data from .csv file """
import pandas as pd
import numpy as np


data = pd.read_csv("data/Project 1 Data.csv")
data = data.dropna().reset_index(drop=True)


## sanity check

# print(data.head()) 
# print(data.columns)
# print(data['X'])
# # data['X'].hist()
# data['Step'].hist()


""" 2.2 Data Visualization - """
import matplotlib.pyplot as plt

# visualizes distribution of count of each value vs value
data.hist() # original bins=10 makes data look weird, fixed w/ 13

plt.figure(figsize=(10,8))
plt.scatter(data['Step'], data['X'])
plt.scatter(data['Step'], data['Y'])
plt.scatter(data['Step'], data['Z'])

plt.xlabel("Step No.")
plt.ylabel("Coordinate Value")

""" 2.3 Correlation Analysis """
import seaborn as sns

plt.figure(figsize=(10,8)) # new figure separate from historgram

corr_matrix = data.corr() # create corrtrix

sns.heatmap(corr_matrix, annot=True) # per corrtrix, strong -ve relationship btwn x and step


""" 2.4 Classification Model Development/Engineering """


""" Data splitting """
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
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

""" Variable Selection """

from sklearn.preprocessing import StandardScaler
# define indep (x) and dep (y) variables
# from previous correlation matrix, strong correlation
# btwn step and x. therefore choose x as dep variable and drop other,
# weaker correlations of Y and Z

x_train = strat_data_train.drop(columns=['Step', 'Y', 'Z'])
y_train = strat_data_train['Step']
x_test = strat_data_test.drop(columns=['Step', 'Y', 'Z'])
y_test = strat_data_test['Step']


""" Data Scaling """

sc = StandardScaler() # define function
sc.fit(x_train) # gets std dev and mean

pd.DataFrame(x_train).to_csv("training_data.csv") # exports original training data
x_train = sc.transform(x_train) # standardizes dataset
pd.DataFrame(x_train).to_csv("scaled_training_data.csv") # saves copy

x_test = sc.transform(x_test) # standardizes dataset


""" Linear Regression Model """
from sklearn.linear_model import LinearRegression, LogisticRegression

model1 = LinearRegression()
model1.fit(x_train, y_train)

y_pred_train1 = model1.predict(x_train)
for i in range(5):
    print("Model 1 Predictions:", y_pred_train1[i], 'Actual Value:',y_train[i])
    
    
""" Logistic Regression Model """

model2 = LogisticRegression()
model2.fit(x_train, y_train)

y_pred_train2 = model2.predict(x_train)
for i in range(5):
    print("Model 2 Predictions:", y_pred_train2[i], 'Actual Value:',y_train[i])
    

""" Random Forest Model """
from sklearn.ensemble import RandomForestRegressor

model3 = RandomForestRegressor()
model3.fit(x_train, y_train)

y_pred_train3 = model3.predict(x_train)
for i in range(5):
    print("Model 3 Predictions:", y_pred_train3[i], 'Actual Value:',y_train[i])
    

 
""" Randomized Search CV """
from sklearn.model_selection import RandomizedSearchCV




# """ 2.5 Model Evaluation """




























