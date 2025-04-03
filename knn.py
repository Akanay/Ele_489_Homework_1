import numpy as np
import pandas as pd
from collections import Counter

#eclidian distance
def euclidean_distance(x_train, x_test_row):
    distances = []
    for i in range(len(x_train)):
        x_train_row = x_train[i]
        current_dist = 0
        for j in range (len(x_train_row)):
            current_dist = current_dist + (x_train_row[j] - x_test_row[j])**2
        distances.append((np.sqrt(current_dist)))
    distances = pd.DataFrame(data=distances, columns=['dist'])
    return distances

#manhattan distance :there was an np.sum function to do all the work for me so I used it this time
def manhattan_distance(x_train, x_test_row):
    distances = np.sqrt(np.sum( abs(x_train - x_test_row), axis=1))
    distances = pd.DataFrame(data=distances, columns=['dist'])
    return distances

#knn algorithm
def my_knn(x_train,x_test,y_train,k):
    y_predict=[]
    for x_test_row in x_test:
        # distance = euclidean_distance(x_train,x_test_row) #row with the euclidean distance
        distance = manhattan_distance(x_train, x_test_row)  #row with the manhattan distance
        sorted_dist = distance.sort_values(by=['dist'], axis=0, ascending=True) # sorting the values according to distance
        kth_nearest = sorted_dist[:k] # choosing the k number of nearest neighbors
        num_of_votes = Counter(y_train.iloc[kth_nearest.index])  # counting function to predict the class
        y_predict_point = num_of_votes.most_common()[0][0]  # taking the most number of votes
        y_predict.append(y_predict_point)  # storing the values in the variable y_predict
    return y_predict
