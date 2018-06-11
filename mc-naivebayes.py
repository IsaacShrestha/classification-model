# Import libraries
import pandas as pd
import numpy as np

# Import dataset
#dataset = pd.read_excel('EventTypeCount.xlsx')
dataset = pd.read_excel('EventCount.xlsx')
X = dataset.iloc[:,1:-1].values
#y = dataset.iloc[:,17].values
y = dataset.iloc[:, 34].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Splitting into train and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = abs(sc_X.fit_transform(X_train))
X_test = abs(sc_X.transform(X_test))
'''

# Fitting multinomial Naive Bayes into train set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the test result
y_pred = classifier.predict(X_test)

# Generating evaluation report
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
Avg_FPR = (FPR0 + FPR1 + FPR2)/3
print(Avg_FPR)




