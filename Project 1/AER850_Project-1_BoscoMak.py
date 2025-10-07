""" AER850 Project 1 - Bosco Mak 501104446 """

""" 2.1 Data processing - read data from .csv file """
import pandas as pd
import numpy as np


data = pd.read_csv("data/Project 1 Data.csv")
data = data.dropna().reset_index(drop=True)
##################################################################################################################
""" 2.2 Data Visualization - """
import matplotlib.pyplot as plt

# visualizes distribution of count of each value vs value
data.hist(bins=13) # original bins=10 makes data look weird, fixed w/ 13

plt.figure(figsize=(10,8))
plt.scatter(data['Step'], data['X'])
plt.scatter(data['Step'], data['Y'])
plt.scatter(data['Step'], data['Z'])

plt.title('Coordinate Value at Step')
plt.xlabel("Step No.")
plt.ylabel("Coordinate Value")
plt.legend(data,loc='best')
plt.show()
##################################################################################################################
""" 2.3 Correlation Analysis """
import seaborn as sns

plt.figure(figsize=(10,8)) # new figure separate from historgram
corr_matrix = data.corr() # create corrtrix
sns.heatmap(corr_matrix, annot=True) # per corrtrix, strong -ve relationship btwn x and step

##################################################################################################################
""" 2.4 Classification Model Development/Engineering """

## Data Splitting
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
# split data based on step value
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

## Variable Selection

from sklearn.preprocessing import StandardScaler

x_train = strat_data_train.drop(columns=['Step'])
y_train = strat_data_train['Step']
x_test = strat_data_test.drop(columns=['Step'])
y_test = strat_data_test['Step']


## Data Scaling

sc = StandardScaler() # define function
sc.fit(x_train) # gets std dev and mean

pd.DataFrame(x_train).to_csv("training_data.csv") # exports original training data
x_train = sc.transform(x_train) # standardizes dataset
pd.DataFrame(x_train).to_csv("scaled_training_data.csv") # saves copy

x_test = sc.transform(x_test) # standardizes dataset

## Model Development    
print("2.4 - Model Development\n_____________________________")

#       Model 1 - Logistic Regression
from sklearn.linear_model import LogisticRegression
print("Model 1 - Logistic Regression\n-----------------")

#Model
model2 = LogisticRegression()
model2.fit(x_train, y_train)

#Prediction
y_pred_train2 = model2.predict(x_train)
for i in range(5):
    print("Model 1 Predictions:", y_pred_train2[i], 'Actual Value:',y_train[i])
    
#Evaluation
from sklearn.metrics import mean_absolute_error
mae_train2 = mean_absolute_error(y_pred_train2, y_train)
print("Model 1 training MAE = ", round(mae_train2,2))

#k-fold cross validation
from sklearn.model_selection import cross_val_score
cv_scores_model2 = cross_val_score(model2, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mae2 = -cv_scores_model2.mean()
print("Model 1 MAE (CV):", round(cv_mae2, 2))

#Pipeline
from sklearn.pipeline import Pipeline
pipeline2 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())])
cv_scores2 = cross_val_score(pipeline2,
                             x_train,
                             y_train,
                             cv=5,
                             scoring='neg_mean_absolute_error')
cv_mae2 = -cv_scores2.mean()
print("Model 1 Pipeline CV MAE:", round(cv_mae2, 2))

pipeline2.fit(x_train, y_train)
y_pred_test2 = pipeline2.predict(x_test)
mae_test2 = mean_absolute_error(y_test, y_pred_test2)
print("Model 1 Pipeline Test MAE:", round(mae_test2, 2))

#Grid Search
print("Grid Search")
from sklearn.model_selection import GridSearchCV, KFold
param_grid1 = {
    'model__penalty': ['l2'],
    'model__C': [0.01, 0.1, 1, 10],
    'model__solver': ['lbfgs', 'liblinear'],
    'model__max_iter': [200, 500]
}

cv = KFold(n_splits=5, shuffle=True, random_state=21)
grid1 = GridSearchCV( # labelling different iterations of 'grid' for future use
    estimator=pipeline2,
    param_grid=param_grid1,
    scoring='neg_mean_absolute_error',
    cv=cv,
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)
grid1.fit(x_train, y_train)

print("Best CV MAE:", -grid1.best_score_)
print("Best params:", grid1.best_params_)
y_pred = grid1.predict(x_test)
print("Test MAE:", mean_absolute_error(y_test, y_pred))


#       Model 2 - Random Forest
from sklearn.ensemble import RandomForestRegressor
print("\nModel 2 - Random Forest\n-----------------")

#Model
model3 = RandomForestRegressor()
model3.fit(x_train, y_train)

#Prediction
y_pred_train3 = model3.predict(x_train)
for i in range(5):
    print("Model 2 Predictions:", y_pred_train3[i], 'Actual Value:',y_train[i])
    
#Evaluation
mae_train3 = mean_absolute_error(y_pred_train3, y_train)
print("Model 2 training MAE = ", round(mae_train3,2))
 
#k-fold cross validation
cv_scores_model3 = cross_val_score(model3, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mae3 = -cv_scores_model3.mean()
print("Model 2 Mean Absolute Error (CV):", round(cv_mae3, 2))

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
print("Model 2 Pipeline CV MAE:", round(cv_mae3, 2))

pipeline3.fit(x_train, y_train)
y_pred_test3 = pipeline3.predict(x_test)
mae_test3 = mean_absolute_error(y_test, y_pred_test3)
print("Model 2 Pipeline Test MAE:", round(mae_test3, 2))

#Grid Search
print("Grid Search")
from sklearn.model_selection import GridSearchCV, KFold
param_grid2 = {
    'model__n_estimators': [10, 30, 50],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2'],
}
cv = KFold(n_splits=5, shuffle=True, random_state=21)
grid2 = GridSearchCV(
    estimator=pipeline3,
    param_grid=param_grid2,
    scoring='neg_mean_absolute_error',
    cv=cv,
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)
grid2.fit(x_train, y_train)

print("Best CV MAE:", -grid2.best_score_)
print("Best params:", grid2.best_params_)
y_pred = grid2.predict(x_test)
print("Test MAE:", mean_absolute_error(y_test, y_pred))


#       Model 3 - SVM
from sklearn import svm
print("\nModel 3 - SVM\n-----------------")

#Model
model4 = svm.SVC()
model4.fit(x_train, y_train)

#Prediction
y_pred_train4 = model4.predict(x_train)
for i in range(5):
    print("Model 3 Predictions:", y_pred_train4[i], 'Actual Value:',y_train[i])
    
#Evaluation
mae_train4 = mean_absolute_error(y_pred_train4, y_train)
print("Model 3 training MAE = ", round(mae_train4,2))
 
#k-fold cross validation
cv_scores_model4 = cross_val_score(model4, x_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mae4 = -cv_scores_model4.mean()
print("Model 3 Mean Absolute Error (CV):", round(cv_mae4, 2))

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
print("Model 3 Pipeline CV MAE:", round(cv_mae4, 2))

pipeline4.fit(x_train, y_train)
y_pred_test4 = pipeline4.predict(x_test)
mae_test4 = mean_absolute_error(y_test, y_pred_test4)
print("Model 3 Pipeline Test MAE:", round(mae_test4, 2))

#Grid Search
print("Grid Search")
from sklearn.model_selection import GridSearchCV, KFold
param_grid3 = {
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'model__kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
    'model__degree': [4],
    'model__gamma': ['scale', 'auto'],
}
cv = KFold(n_splits=5, shuffle=True, random_state=21)
grid3 = GridSearchCV(
    estimator=pipeline4,
    param_grid=param_grid3,
    scoring='neg_mean_absolute_error',
    cv=cv,
    n_jobs=-1,
    refit=True,           
    verbose=1,
    return_train_score=True
)
grid3.fit(x_train, y_train)

print("Best CV MAE:", -grid3.best_score_)
print("Best params:", grid3.best_params_)
y_pred = grid3.predict(x_test)
print("Test MAE:", mean_absolute_error(y_test, y_pred))

#RandomizedSearch Cross Validation
print("\nRandomizedSearchCV")
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
##################################################################################################################
""" 2.5 Model Performance Analysis """
print("\n2.5 - Model Performance Analysis\n_____________________________")

#Logistic Regression Analysis
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

print("LogisticRegression Metrics")
clf1 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=21))
])

clf1.set_params(**grid1.best_params_) # ensures best parameters
clf1.fit(x_train, y_train)
print("LG Training accuracy:", clf1.score(x_train, y_train))
print("LG Test accuracy:", clf1.score(x_test, y_test))

y_pred_clf1 = clf1.predict(x_test)
cm_clf1 = confusion_matrix(y_test, y_pred_clf1)
print("LG Confusion Matrix:")
print(cm_clf1)
precision_clf1 = precision_score(y_true = y_test,
                                 y_pred = y_pred_clf1,
                                 average='weighted')
recall_clf1 = recall_score(y_test, y_pred_clf1, average='weighted')
f1_clf1 = f1_score(y_test, y_pred_clf1, average='weighted')
print("LG Precision:", precision_clf1)
print("LG Recall:", recall_clf1)
print("LG F1 Score:", f1_clf1)


#RandomForestRegressor Analysis
print("\nRandomForest Metrics")
from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier(n_estimators=200, random_state = 21)
clf2.fit(x_train, y_train)
print("RF Training accuracy:", clf2.score(x_train, y_train))
print("RF Test accuracy:", clf2.score(x_test, y_test))

y_pred_clf2 = clf2.predict(x_test)
cm_clf2 = confusion_matrix(y_test, y_pred_clf2)
print("RF Confusion Matrix:")
print(cm_clf2)
precision_clf2 = precision_score(y_test, y_pred_clf2, average='weighted')
recall_clf2 = recall_score(y_test, y_pred_clf2, average='weighted')
f1_clf2 = f1_score(y_test, y_pred_clf2, average='weighted')
print("RF Precision:", precision_clf2)
print("RF Recall:", recall_clf2)
print("RF F1 Score:", f1_clf2)

#SVM Analysis
print("\nSVM Metrics")
clf3 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", svm.SVC(probability=True, random_state=21))
])

clf3.set_params(**rand.best_params_) # ensures use of best parameters when evaluating
clf3.fit(x_train, y_train)
print("SVM Training accuracy:", clf3.score(x_train, y_train))
print("SVM Test accuracy:", clf3.score(x_test, y_test))

y_pred_clf3 = clf3.predict(x_test)
cm_clf3 = confusion_matrix(y_test, y_pred_clf3)
print("SVM Confusion Matrix:")
print(cm_clf3)
precision_clf3 = precision_score(y_true = y_test,
                                 y_pred = y_pred_clf3,
                                 average='weighted')
recall_clf3 = recall_score(y_test, y_pred_clf3, average='weighted')
f1_clf3 = f1_score(y_test, y_pred_clf3, average='weighted')
print("SVM Precision:", precision_clf3)
print("SVM Recall:", recall_clf3)
print("SVM F1 Score:", f1_clf3)
##################################################################################################################
""" 2.6 - Stacked Model Performance"""
print("\n\n2.6 - Stacked Model Performance\n_____________________________")
from sklearn.ensemble import StackingClassifier

estimators = [
    ('clf2', clf2),
     ('clf3', clf3)
]

stack_clf = StackingClassifier(
    estimators = estimators,
    final_estimator = LogisticRegression(max_iter = 1000, random_state = 21)
)

stack_clf.fit(x_train, y_train)
print("Stacked Classifier Training Accuracy: ",stack_clf.score(x_train,y_train))
print("Stacked Classifier Test Accuracy: ", stack_clf.score(x_test, y_test))
    
y_pred_stack = stack_clf.predict(x_test)
cm_stack_clf = confusion_matrix(y_test, y_pred_stack)
print("Stacked Classifier Confusion Matrix:")
print(cm_stack_clf)
precision_stack = precision_score(y_test, y_pred_stack, average='weighted')
recall_stack = recall_score(y_test, y_pred_stack, average='weighted')
f1_stack = f1_score(y_test, y_pred_stack, average='weighted')
print("Stacked Classifier Precision:", precision_stack)
print("Stacked Classifier Recall:", recall_stack)
print("Stacked Classifier F1 Score:", f1_stack)
##################################################################################################################
""" 2.7 Model Evaluation """
print("\n\n2.7 Model Evaluation\n_____________________________")
import joblib

joblib.dump(stack_clf, "project1_testfile.joblib")

final_clf = joblib.load("project1_testfile.joblib")

a = [9.375,3.0625,1.51]
b = [6.995,5.125,0.3875]
c = [0,3.0625,1.93]
d = [9.4,3,1.8]
e = [9.4,3,1.3]

eval_data = np.array([a,b,c,d,e])
pred = final_clf.predict(eval_data)
print(pred)    