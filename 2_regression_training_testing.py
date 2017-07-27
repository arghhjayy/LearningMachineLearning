import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

#Get all the columns of Google's stock
#Basics:
#features: input, labels: output
df = quandl.get('WIKI/GOOGL')

#Get only the necessary columns
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#Generate a 'high (minus) low percentage' column
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100

#Generate a 'percentage change' column
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

#Further filter the columns
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#The column name which is to be forecasted
forecast_col = 'Adj. Close'

#Fill the NA(Not Available) columns with -99999
df.fillna(-99999, inplace=True)

#                             1 percent of the data
#                             ^
forecast_out = int(math.ceil(0.01*len(df)))

print('Predict what will happen in ' + str(forecast_out) + ' days.')

df['label'] = df[forecast_col].shift(-forecast_out)

#Delete NaN(Not a number) fields
#This will discard all the features which fall into the future
df.dropna(inplace=True)

#X is a feature: input, all the columns except 'label'
X = np.array(df.drop(['label'], 1))

#y is a label: output, a column named 'label'
y = np.array(df['label'])

#Scale the features to a smaller range of values
X = preprocessing.scale(X)

#Shuffles up data and returns random elements from the
#given data to X_train, X_test, y_train, y_test respectively
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#Create a classifier (in this case, a LinearRegression classifier
#is used
clf = LinearRegression()

#Find what would the m and c be in the following equation:
#y = mx + c
#which is an equation of a straight line: a LINEAR fit
clf.fit(X_train, y_train)

#Find the score/accuracy against some testing data: X_test, y_test
accuracy = clf.score(X_test, y_test)

#Print the accuracy in percentage
print('Prediction accuracy: ' + str(accuracy * 100) + '%')