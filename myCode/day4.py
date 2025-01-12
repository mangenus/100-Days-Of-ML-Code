import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



data = pd.read_csv('./datasets/Social_Network_Ads.csv')

X = data.iloc[:,[2,3]].values
Y = data.iloc[:,4].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train,Y_train)

pred = classifier.predict(X_test)
cm = confusion_matrix(Y_test, pred)

c = classifier.coef_

X_test_true = X_test[pred]
print(X_test)