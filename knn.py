# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 17:18:47 2018

@author: IsaacShrestha
K - Nearest Neighbour
"""

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
#dataset = pd.read_excel('EventtypeCount.xlsx')
dataset = pd.read_excel('EventCount.xlsx') 
X = dataset.iloc[:,1:-1].values
#y = dataset.iloc[:,17].values
y = dataset.iloc[:, 34].values

# Encoding the categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Split the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

'''
# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# Predicting the Test set result
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print cr

# Calculating average precision score
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)

# Calculating false positives and fase negatives
TP = float(cm[0][0])
FP = float(cm[1][0])
FN = float(cm[0][1])
TN = float(cm[1][1])

FPR = float((FP/(FP+TN))*100)
FNR = float((FN/(TP+FN))*100)
