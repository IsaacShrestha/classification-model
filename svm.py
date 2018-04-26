#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 01:16:14 2018

@author: isaacshrestha
"""

# Support Vector Machine

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_excel('dataset.xlsx')
X = dataset.iloc[:, [4,5,6,9,10]].values
y = dataset.iloc[:, 12].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# Making all datatype of event and size as string for encoding
X[:, 2] = labelencoder_X.fit_transform(X[:,2].astype(str))
X[:, 3] = labelencoder_X.fit_transform(X[:,3].astype(str))

X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])

onehotencoder = OneHotEncoder(categorical_features = [1,2,3,4])
X = onehotencoder.fit_transform(X).toarray()

# Encoding dependent variable in y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into Training and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculating accuracy 
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# Calculating average precision score
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)