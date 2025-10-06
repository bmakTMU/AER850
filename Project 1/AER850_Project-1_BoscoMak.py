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
plt.legend(loc='best')

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


""" Model 1 - Linear Regression Model """
from sklearn.linear_model import LinearRegression
print("-----------------")


#Model
model1 = LinearRegression()
model1.fit(x_train, y_train)

#Prediction
y_pred_train1 = model1.predict(x_train)
for i in range(5):
    print("Model 1 Predictions:", y_pred_train1[i], 'Actual Value:',y_train[i])
    
#Evaluation
from sklearn.metrics import mean_absolute_error
mae_train1 = mean_absolute_error(y_pred_train1, y_train)
print("Model 1 training MAE = ", round(mae_train1,2))

#k-fold cross validation
from sklearn.model_selection import cross_val_score
cv_scores_model1 = cross_val_score(model1, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mae1 = -cv_scores_model1.mean()
print("Model 1 MAE (CV):", round(cv_mae1, 2))
    
#Pipeline
from sklearn.pipeline import Pipeline
pipeline1 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())])
cv_scores1 = cross_val_score(pipeline1,
                             x_train,
                             y_train,
                             cv=5,
                             scoring='neg_mean_absolute_error')
cv_mae1 = -cv_scores1.mean()
print("Model 1 Pipeline CV MAE:", round(cv_mae1, 2))

pipeline1.fit(x_train, y_train)
y_pred_test1 = pipeline1.predict(x_test)
mae_test1 = mean_absolute_error(y_test, y_pred_test1)
print("Model 1 Pipeline Test MAE:", round(mae_test1, 2))



    
""" Model 2 - Logistic Regression Model """
from sklearn.linear_model import LogisticRegression
print("-----------------")


#Model
model2 = LogisticRegression()
model2.fit(x_train, y_train)

#Prediction
y_pred_train2 = model2.predict(x_train)
for i in range(5):
    print("Model 2 Predictions:", y_pred_train2[i], 'Actual Value:',y_train[i])
    
#Evaluation
mae_train2 = mean_absolute_error(y_pred_train2, y_train)
print("Model 2 training MAE = ", round(mae_train2,2))

#k-fold cross validation
cv_scores_model2 = cross_val_score(model2, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mae2 = -cv_scores_model2.mean()
print("Model 2 MAE (CV):", round(cv_mae2, 2))

#Pipeline
pipeline2 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())])
cv_scores2 = cross_val_score(pipeline2,
                             x_train,
                             y_train,
                             cv=5,
                             scoring='neg_mean_absolute_error')
cv_mae2 = -cv_scores2.mean()
print("Model 2 Pipeline CV MAE:", round(cv_mae1, 2))

pipeline2.fit(x_train, y_train)
y_pred_test2 = pipeline2.predict(x_test)
mae_test2 = mean_absolute_error(y_test, y_pred_test2)
print("Model 2 Pipeline Test MAE:", round(mae_test2, 2))

#Grid Search
print("Grid Search")
from sklearn.model_selection import GridSearchCV, KFold
param_grid = {
    'model__penalty': ['l2'],
    'model__C': [0.01, 0.1, 1, 10],
    'model__solver': ['lbfgs', 'liblinear'],
    'model__max_iter': [200, 500]
}

cv = KFold(n_splits=5, shuffle=True, random_state=21)
grid = GridSearchCV(
    estimator=pipeline2,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=cv,
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)
grid.fit(x_train, y_train)

print("Best CV MAE:", -grid.best_score_)
print("Best params:", grid.best_params_)
y_pred = grid.predict(x_test)
print("Test MAE:", mean_absolute_error(y_test, y_pred))


""" Model 3 Random Forest Model """
from sklearn.ensemble import RandomForestRegressor
print("-----------------")


#Model
model3 = RandomForestRegressor()
model3.fit(x_train, y_train)

#Prediction
y_pred_train3 = model3.predict(x_train)
for i in range(5):
    print("Model 3 Predictions:", y_pred_train3[i], 'Actual Value:',y_train[i])
    
#Evaluation
mae_train3 = mean_absolute_error(y_pred_train3, y_train)
print("Model 3 training MAE = ", round(mae_train3,2))
 
#k-fold cross validation
cv_scores_model3 = cross_val_score(model3, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mae3 = -cv_scores_model3.mean()
print("Model 3 Mean Absolute Error (CV):", round(cv_mae3, 2))

#Pipeline
pipeline3 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())])
cv_scores3 = cross_val_score(pipeline3,
                             x_train,
                             y_train,
                             cv=5,
                             scoring='neg_mean_absolute_error')
cv_mae3 = -cv_scores3.mean()
print("Model 3 Pipeline CV MAE:", round(cv_mae3, 2))

pipeline3.fit(x_train, y_train)
y_pred_test3 = pipeline3.predict(x_test)
mae_test3 = mean_absolute_error(y_test, y_pred_test3)
print("Model 3 Pipeline Test MAE:", round(mae_test3, 2))

#Grid Search
print("Grid Search")
from sklearn.model_selection import GridSearchCV, KFold
param_grid = {
    'model__n_estimators': [10, 30, 50],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2'],
}
cv = KFold(n_splits=5, shuffle=True, random_state=21)
grid = GridSearchCV(
    estimator=pipeline3,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=cv,
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)
grid.fit(x_train, y_train)

print("Best CV MAE:", -grid.best_score_)
print("Best params:", grid.best_params_)
y_pred = grid.predict(x_test)
print("Test MAE:", mean_absolute_error(y_test, y_pred))



""" Model 4 - SVM Model """
from sklearn import svm
print("-----------------")

#Model
model4 = svm.SVC()
model4.fit(x_train, y_train)

#Prediction
y_pred_train4 = model4.predict(x_train)
for i in range(5):
    print("Model 4 Predictions:", y_pred_train4[i], 'Actual Value:',y_train[i])
    
#Evaluation
mae_train4 = mean_absolute_error(y_pred_train4, y_train)
print("Model 4 training MAE = ", round(mae_train4,2))
 
#k-fold cross validation
cv_scores_model4 = cross_val_score(model4, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mae4 = -cv_scores_model4.mean()
print("Model 4 Mean Absolute Error (CV):", round(cv_mae4, 2))

#Pipeline
pipeline4 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', svm.SVC())])
cv_scores4 = cross_val_score(pipeline4,
                             x_train,
                             y_train,
                             cv=5,
                             scoring='neg_mean_absolute_error')
cv_mae4 = -cv_scores4.mean()
print("Model 4 Pipeline CV MAE:", round(cv_mae4, 2))

pipeline4.fit(x_train, y_train)
y_pred_test4 = pipeline4.predict(x_test)
mae_test4 = mean_absolute_error(y_test, y_pred_test4)
print("Model 4 Pipeline Test MAE:", round(mae_test4, 2))


#Grid Search
print("Grid Search")
from sklearn.model_selection import GridSearchCV, KFold
param_grid = {
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'model__kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
    'model__degree': [4],
    'model__gamma': ['scale', 'auto'],
}
cv = KFold(n_splits=5, shuffle=True, random_state=21)
grid = GridSearchCV(
    estimator=pipeline4,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=cv,
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)
grid.fit(x_train, y_train)

print("Best CV MAE:", -grid.best_score_)
print("Best params:", grid.best_params_)
y_pred = grid.predict(x_test)
print("Test MAE:", mean_absolute_error(y_test, y_pred))

#RandomizedSearch Cross Validation
print("RandomizedSearchCV")
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'model__kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
    'model__degree': [4],
    'model__gamma': ['scale', 'auto'],
}
cv = KFold(n_splits=5, shuffle=True, random_state=21)
rand = RandomizedSearchCV(
    estimator=pipeline4,
    param_distributions=param_distributions,
    scoring='neg_mean_absolute_error',
    cv=cv,
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)
rand.fit(x_train, y_train)

print("Best CV MAE:", -rand.best_score_)
print("Best params:", rand.best_params_)
y_pred = rand.predict(x_test)
print("Test MAE:", mean_absolute_error(y_test, y_pred))

""" 2.5 Model Evaluation """
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

clf1 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=21))
])

clf1.fit(x_train, y_train)
print("Training accuracy:", clf1.score(x_train, y_train))
print("Test accuracy:", clf1.score(x_test, y_test))

y_pred_clf1 = clf1.predict(x_test)
cm_clf1 = confusion_matrix(y_test, y_pred_clf1)
print("Confusion Matrix:")
print(cm_clf1)
precision_clf1 = precision_score(y_test, y_pred_clf1)
recall_clf1 = recall_score(y_test, y_pred_clf1)
f1_clf1 = f1_score(y_test, y_pred_clf1)
print("Precision:", precision_clf1)
print("Recall:", recall_clf1)
print("F1 Score:", f1_clf1)


