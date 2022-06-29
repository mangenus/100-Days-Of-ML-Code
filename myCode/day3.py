import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('../datasets/50_startups.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values


onehotencoder = OneHotEncoder(categories='auto')
Z = onehotencoder.fit_transform(X[:,3].reshape(-1,1)).toarray()[:,1:]

X = np.append(X[:,:-1],Z,axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_test)

print(Y_pred)