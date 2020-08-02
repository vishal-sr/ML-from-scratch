import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import warnings
import random

data = pd.read_csv('breast-cancer-wisconsin-data.txt')

# Data Preprocessing.
data.replace('?', -99999, inplace = True)
data.drop(['id'], axis = 1, inplace = True)
data = data.astype(float).values
random.shuffle(data)

test_size = 1/4
train_data = {2 : [], 4 : []}
test_data = {2 : [], 4 : []}
train = data[ : -int(test_size * len(data))]  # shape - (466, 10) fetching the data(s) from the main dataset (0 to test_size).
                                              # suppose, the size of the dataset is 100 and the test_size is 1/4 (0.25)
                                              # test_size * len(data) = 0.25 * 100 = 25 
                                              # So, here data[ : -int(test_size * len(data))] represents the first 75 data(s)
test = data[-int(test_size * len(data)) : ]  # shape - (233, 10) featching the remaining data(s) from test_size to last record
                                             # suppose, the size of the dataset is 100 and the test_size is 1/4 (0.25)
                                             # test_size * len(data) = 0.25 * 100 = 25 
                                             # So, here data[-int(test_size * len(data)) : ] represents the last 25 data(s)

for i in train:
    train_data[i[-1]].append(i[:-1])# train_data consist of two keys 2 and 4
                                    # i[-1] is the last column which is 'class' column of the dataset which can only be 2 or 4
                                    # if i[-1] is 2 then except last column all other columns is appended to train_data of key 2
                                    # same for i[-1] is 4
    
for i in test:
    test_data[i[-1]].append(i[:-1])# test_data consist of two keys 2 and 4
                                   # i[-1] is the last column which is 'class' column of the dataset which can only be 2 or 4
                                   # if i[-1] is 2 then except last column all other columns is appended to test_data of key 2
                                   # same for i[-1] is 4

# Algorithm.
def k_nearest_neighbors(data, predict, k = 5): # k - number of neighbors.
    if len(data) >= k:
        warnings.warn('The data consist of more classes than K...Please set the K value greater than the number of classes...')
        
    distances = [] # setting empty list to store all the distances
    
    for labels in data:
        for coordinates in data[labels]:
            euclidean_distance = np.linalg.norm(np.array(coordinates) - np.array(predict)) #calculating Euclidean distance
            distances.append((euclidean_distance, labels)) #appending each distances, labels to the list.
            
    groups = [d[1] for d in sorted(distances)[:k]]           # for d in sorted(distances)[:k]:
                                                             #     groups = d[1]
    group = Counter(groups).most_common(1)[0][0]  # fetching the more occurence group/class       
    confidence = Counter(groups).most_common(1)[0][1] / k  # how likely the patient belongs to some group/class (probability) 
    
    return group, confidence

correct = 0  #number of correct predictions
total = 0  #number of total predictions

for labels in test_data:  # Runs number of classes times.
    for coordinates in test_data[labels]:  #Runs number of data inside each class.
        
        group, confidence = k_nearest_neighbors(train_data, coordinates, k = 5) # Passing the training data and new datapoints 
                                                                                # to the K-NN algorithm.
                                                                            # coordinates is the new datapoint to be predicted.
        if labels == group:    # group is the prediction of new datapoint (predicted class) which is compared with real class
                               # of the new datapoint.
                
            correct += 1       # if both are same then correct is added by 1
            
        else:
            print(confidence)  # if both are not same then print the probability of incorrectness in prediction
            
        total += 1
        
print("Accuracy = ", correct / total, "%") # Accuracy of the model
