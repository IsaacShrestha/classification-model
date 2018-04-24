#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:14:51 2018

@author: isaacshrestha
"""
# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Importing the dataset
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


