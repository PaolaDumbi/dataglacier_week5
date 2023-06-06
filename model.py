# Importing the libraries
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('/Users/paoladumbi/Documents/PROJECTS/SUMMER/week4/ads.csv')
dataset.reset_index(drop=True)
X = dataset.iloc[:, :3]
y = dataset.iloc[:, -1]
print(y.head(6))

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('/Users/paoladumbi/Documents/PROJECTS/SUMMER/week4/model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('/Users/paoladumbi/Documents/PROJECTS/SUMMER/week4/model.pkl','rb'))
# print(model.predict([[192.5, 34.12, 45.01]]))
