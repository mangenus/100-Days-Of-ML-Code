import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('../datasets/studentscores.csv')

X = dataset.iloc[:,:1].values
Y = dataset.iloc[:,1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/4, random_state = 0)

regressor = LinearRegression()
regressor = regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_test)

plt.figure(1)
plt.scatter(X_train,Y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')

plt.figure(2)
plt.scatter(X_test, Y_test, color = 'green')
plt.plot(X_test, Y_pred, color='black')

plt.show()