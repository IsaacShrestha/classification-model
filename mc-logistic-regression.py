# Importing the libraries
import pandas as pd
import numpy as np

# import the dataset
#dataset = pd.read_excel('EventTypeCount.xlsx')
dataset = pd.read_excel('EventCount.xlsx')
X = dataset.iloc[:,1:-1].values
#y = dataset.iloc[:,17].values
y = dataset.iloc[:,34].values

# Encoding the categorical data
from sklearn.preprocessing import LabelEncoder
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Splitting datawset into train and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting logistic regression to train set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluating classifier through confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print cr