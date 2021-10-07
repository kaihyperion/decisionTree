import numpy as np 
import os
import csv

def load_data(data_path):
    f = open(data_path, 'r')
    reader = csv.reader(f, delimiter=',')  # csv.reader() reads the file
    # Get all but last element TYPE: List of String next() also returns all but header
    attribute_names = next(reader)[:-1]
    data = np.array(list(reader)).astype(float)  # returns array of ints **IF i want a more robut use i can have it be strings

    features = data[: , 0:-1] #This should index out the features portion of N X K Matrix
    targets = data[: , -1] #This should index out only the last column of 1 x N Matrix for Class attribute data
    return features, targets, attribute_names



def train_test_split(features, targets, fraction):
    if (fraction > 1.0):
        raise ValueError('N cannot be bigger than number of examples!')

    N = int(features.shape[0] * fraction)
    train_features = features[:N]
    train_targets = targets[:N]
    test_features = features[N:]
    test_targets = targets[N:]
    return train_features, train_targets, test_features, test_targets

