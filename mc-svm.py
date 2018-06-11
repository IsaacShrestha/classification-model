# Import libraries
import pandas as pd
import numpy as np

# Import the dataset
#dataset = pd.read_excel('EventCount.xlsx')
dataset = pd.read_excel('EventTypeCount.xlsx')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,17].values
#y = dataset.iloc[:,34].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Split dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=60, test_size=15, random_state=0)

'''
# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

# Fitting One-Vs-One SVM to the training set
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
clf = OneVsOneClassifier(LinearSVC(random_state=0))
clf.fit(X_train, y_train)

# Predicting the result
y_pred = clf.predict(X_test)

# Developing confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print(cr)

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
FNR0 = FN0 / (FN0+TP0)
FNR1 = FN1 / (FN1+TP1)
FNR2 = FN2 / (FN2+TP2)

Avg_FPR = (FPR0 + FPR1 + FPR2)/3
print( Avg_FPR)

Avg_FNR = (FNR0 + FNR1 + FNR2) / 3
print(Avg_FNR)