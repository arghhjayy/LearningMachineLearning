import numpy as np
import warnings
import pandas as pd
from collections import Counter
import random

def k_nearest_neighbors(data, predict_for, k = 3):
    '''

    Calculates and returns the predicted type for a data sample

    data: (80%) of the overall data on which the algorithm is trained
    predict_for: a sample of data for which the prediction is to be done
    k: number of nearest elements to be found at least for the sample to be of that group

    '''

    #If the number of groups in the data passed is greater than k, warn the user
    #In our case:
    #len(data) = 2 (two types of tumours)
    #k = 3 for default and 5 (passed below)
    if len(data) >= k:
        warnings.warn('k is set to a value less than the total voting groups, boi')

    #All the distances from the point
    distances = []

    #for every group, i. e. 2 and 4
    for group in data:
        #for every patient with 2 and 4 type tumour
        for features in data[group]:
            #Calculate the Euclidean distance (fancy)
            euclidean_distance = np.linalg.norm(np.array(features) - np.array((predict_for)))
            #Add a list of a combination of Euclidean distance and the group
            #identification to the list 'distance'
            distances.append([euclidean_distance, group])

    #Get the votes for every possible point
    #Only collect atmost k points, because 'k nearest neighbors'
    votes = [i[1] for i in sorted(distances)[:k]]

    #Calculate the most votes' group
    vote_result = Counter(votes).most_common(1)[0][0]

    #Return the most votes' group
    return vote_result

#Open the data file
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
#Replace the occurrence of a '?' by -99999
df.replace('?', -99999, inplace=True)
#Delete the id column
df.drop(['id'], 1, inplace=True)

#Convert the data to a float form from string form
full_data = df.astype(float).values.tolist()

#Shuffle the data
random.shuffle(full_data)

#0.2 = 20% of the data will be used for testing
#80% will be used for training
test_size = 0.2

#training data set: one for malignant(2) and benign(4)
train_set = {2: [], 4: []}
#testing data set: one for malignant(2) and benign(4)
test_set = {2: [], 4: []}

#Use 80% of the data set for training
train_data = full_data[:-int(test_size * len(full_data))]
#Use 20% for testing
test_data = full_data[-int(test_size * len(full_data)):]

#Add the data to the respective dictionaries
for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

#Counters for measuring accuracy:
#correct for every correct prediction
correct = 0
#total for each sample taken
total = 0

#For every group i. e. for 2 and 4 (malignant and benign)
for group in test_set:
    #For every entry in the group
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        #If prediction is right, increment correct by 1
        if group == vote:
            correct += 1
        #Increment total number of samples processed till now
        #by one
        total += 1

#Display the accuracy percentage
print('Accuracy: ', correct/total * 100, '%')