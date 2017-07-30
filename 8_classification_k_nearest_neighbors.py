import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

#Read the breast cancer data using pandas
#Basics:
#Input: Various parameters of a lump on a breast like lump size, texture, etc (score out of 10)
#Output: Is the tumour malignant or benign (2 or 4)
df = pd.read_csv('breast-cancer-wisconsin.data.txt', skipinitialspace=True)

#Replace every empty field with a -99999
df.replace('?', -99999, inplace=True)

#Remove the id column as it is irrelevant
df.drop(['id'], 1, inplace=True)

#Build input: all the columns except the 'class' column
X = np.array(df.drop(['class'], 1))
#Build output: only the 'class' column
y = np.array(df['class'])

#Get data for training and testing randomly
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#Create a K Nearest Neighbors classifiers
clf = neighbors.KNeighborsClassifier()

#Train the algorithm by training data
clf.fit(X_train, y_train)

#Run the algorithm on test data and get the accuracy
accuracy = clf.score(X_test, y_test)
print(accuracy)

#Calculate decision for
example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 1, 2, 2, 3, 2, 1]])

#For fixing the deprecation warning            in our case = 2
example_measures = example_measures.reshape(len(example_measures), -1)

#Predict the result for the examples given in example_measures(both of them)
prediction = clf.predict(example_measures)
print(prediction)