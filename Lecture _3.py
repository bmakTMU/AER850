import pandas as pd

data = pd.read_csv("data/housing.csv")

print(data.head()) #
print(data.columns) # prints names of columns
print(data['ocean_proximity']) # returns list of values under col "ocean proximity"

# how to view data? -> use histograms
# adding a .hist() to end of "data" will give histogram (in plots field)

data['ocean_proximity'].hist()
data.hist()

# how do we make the 'ocean proximity' data numerical?
# -> use SciKit-learn

from sklearn.preprocessing import OneHotEncoder

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