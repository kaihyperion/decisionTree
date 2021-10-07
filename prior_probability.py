import numpy as np

class PriorProbability():
    def __init__(self):
        
        self.most_common_class = None
        self.positive_class = 1
        self.negative_class = 0

    def fit(self, features, targets):
        
        #Training model
        pos = 0
        neg = 0
        for i in targets:
            if i == 1:
                pos += 1
            else:
                neg += 1
        if pos >= neg:
            self.most_common_class = self.positive_class
        else:
            self.most_common_class = self.negative_class


    def predict(self, data):
        #Using the training to the new data
        # Initialize a prediction array thats the size of the num of examples
        #Return whatever the most common class is
        # if my most common class is 0, You want to return a prediction array
        # which is the size of the # of examples that you want to return
        #an array thats filled with the most common class. Thats how the learner predicts
        #iterate through the list data shape []
        
        
        if self.most_common_class == True:
            predictions = np.ones(data.shape[0])
        else:
            predictions = np.zeros(data.shape[0])

        return predictions
