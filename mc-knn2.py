# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 13:15:08 2018

@author: IsaacShrestha
mc-knn
"""
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import dataset
'''
train_dataset = pd.read_excel('Train-EventTypeCount.xlsx')
test_dataset = pd.read_excel('Test-EventTypeCount.xlsx')
X_train = train_dataset.iloc[:60,1:-1].values
y_train = train_dataset.iloc[:60,17].values
X_test = test_dataset.iloc[:,1:-1].values
y_test = test_dataset.iloc[:,17].values
'''
train_dataset = pd.read_excel('Train-EventCount.xlsx')
test_dataset = pd.read_excel('Test-EventCount.xlsx')
X_train = train_dataset.iloc[:60,1:-1].values
y_train = train_dataset.iloc[:60,34].values
X_test = test_dataset.iloc[:,1:-1].values
y_test = test_dataset.iloc[:,34].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelEncoder_y = LabelEncoder()
labelEncoder_y_test = LabelEncoder()
y_train = labelEncoder_y.fit_transform(y_train)
y_test = labelEncoder_y_test.fit_transform(y_test)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

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

# Calculating FP and FPR
TP0 = float(cm[0][0])
TP1 = float(cm[1][1])
TP2 = float(cm[2][2])
FN0 = float(cm[0][1])+float(cm[0][2])
FN1 = float(cm[1][0])+float(cm[1][2])
FN2 = float(cm[2][0])+float(cm[2][1])
FP0 = float(cm[1][0])+float(cm[2][0])
FP1 = float(cm[0][1])+float(cm[2][1])
FP2 = float(cm[0][2])+float(cm[1][2])
TN0 = float(cm[1][1])+float(cm[1][2])+float(cm[2][1])+float(cm[2][2])
TN1 = float(cm[0][0])+float(cm[0][2])+float(cm[2][0])+float(cm[2][2])
TN2 = float(cm[0][0])+float(cm[0][1])+float(cm[1][0])+float(cm[1][1])
FPR0 = FP0 / (FP0+TN0)
FPR1 = FP1 / (FP1+TN1)
FPR2 = FP2 / (FP2+TN2)
Avg_FPR = (FPR0 + FPR1 + FPR2)/3
print(Avg_FPR)
precision = (TP0+TP1+TP2)/(TP0+TP1+TP2+FP0+FP1+FP2)
print(precision)
