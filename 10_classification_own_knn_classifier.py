from math import sqrt
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import  style
from collections import Counter

#Use this style to plot the graph
style.use('fivethirtyeight')

#Dataset has two groups:
#group k and group r
#Plotting the points [1, 2], [2, 3], [3, 1] and [6, 5], [7, 7] and [8, 6] on a graph
#will make it easier to understand the groups
dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}

#Which group does this point belong to?
new_features = [5, 7]

def k_nearest_neighbors(data, predict_for, k = 3):
    '''

    Given training data(data),
    find the group to which the point (predict_for) belongs to

    '''
    if len(data) >= k:
        warnings.warn('k is set to a value less than the total voting groups, boi')

    #All of them distances
    distances = []

    #For each group i. e. for k and r
    for group in data:
        #For each of their points
        for features in data[group]:
            #Calculate the distance of the point belonging to group x(k or r) from the 'predict_for' point
            euclidean_distance = np.linalg.norm(np.array(features) - np.array((predict_for)))
            #Add this distance to the list of distances along with that group signature(k or r)
            distances.append([euclidean_distance, group])

    #Find the group of the first k points which are closest to the 'predict_for' point
    votes = [i[1] for i in sorted(distances)[:k]]
    #Calculate which group had the maximum of the votes
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

#Calculate group for [5, 7] using the dataset
result = k_nearest_neighbors(dataset, new_features)
print('New feature belongs to ' + result[0] + ' group')

#Plotting it will make it clear, yep
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]

plt.scatter(new_features[0], new_features[1], s=100, color=result)

plt.show()