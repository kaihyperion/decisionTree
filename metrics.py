import numpy as np

def confusion_matrix(actual, predictions):
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
        
    for i in range(actual.shape[0]):

        if actual[i] == 1:  #True_...
            if predictions[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if predictions[i] == 0:
                tn += 1
            else:
                fp += 1

    confusion_matrx = np.array([
                        [tn, fp],
                        [fn, tp]
                        ])
    return confusion_matrx

def accuracy(actual, predictions):
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
        
    matrix = confusion_matrix(actual, predictions)
    
    #Accuray = TN + TP / (TN + FP + FN + TP)
    accuracy = (matrix[0,0] + matrix[1,1]) / matrix.sum(dtype='float')
    
    return accuracy

def precision_and_recall(actual, predictions):
    
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    matrix = confusion_matrix(actual, predictions)
    #Precision = TP / TP + FP
    precision = matrix[1,1] / matrix.sum(axis=0, dtype='float')[-1]

    #RECALL = TP / TP+FN
    recall = matrix[1,1] / matrix.sum(axis=1, dtype='float')[-1]

    return precision, recall

def f1_measure(actual, predictions):
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
        
    precision, recall = precision_and_recall(actual, predictions)
 
    F1_measure = (2 * recall * precision)/ (recall + precision)
    return F1_measure

