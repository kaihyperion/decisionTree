from .decision_tree import DecisionTree
from .prior_probability import PriorProbability
from .metrics import precision_and_recall, confusion_matrix, f1_measure, accuracy
from .data import load_data, train_test_split

def run(data_path, learner_type, fraction):
    """
        1. takes in a path to a dataset
        2. loads it into a numpy array with `load_data`
        3. instantiates the class used for learning from the data using learner_type (e.g
           learner_type is 'decision_tree', 'prior_probability')
        4. splits the data into training and testing with `train_test_split` and `fraction`.
        5. trains a learner using the training split with `fit`
        6. tests the trained learner using the testing split with `predict`
        7. evaluates the trained learner with precision_and_recall, confusion_matrix, and
           f1_measure

    Args:
        data_path (str): path to csv file containing the data
        learner_type (str): either 'decision_tree' or 'prior_probability'. For each of these,
            the associated learner is instantiated and used for the experiment.
        fraction (float between 0.0 and 1.0): fraction of examples to be drawn for training

    Returns:
        confusion_matrix (np.array): Confusion matrix of learner on testing examples
        accuracy (np.float): Accuracy on testing examples using learner
        precision (np.float): Precision on testing examples using learner
        recall (np.float): Recall on testing examples using learner
        f1_measure (np.float): F1 Measure on testing examples using learner
    """

    #1 Take in a path to dataset and use load_data from .data.py

    features, targets, attribute_names = load_data(data_path)

    #2 Initiates the class used for learning from the data using learner_type
    # (.e.g learner_type is decision_tree', 'prior_probability')

    if fraction == 1.0:
        training_features = testing_features = features
        training_targets = testing_targets = targets
    else:
        training_features, training_targets, testing_features, testing_targets \
            = train_test_split(features, targets, fraction)

    if learner_type == "prior_probability":
        PriorProbability().fit(training_features, training_targets)
        prediction = PriorProbability().predict(testing_features)
    else:
        DecisionTree(attribute_names).fit(training_features, training_targets)
        prediction = DecisionTree(attribute_names).predict(testing_features)

    confuscious_matrix = confusion_matrix(testing_targets, prediction)
    accuraC = accuracy(testing_targets, prediction)
    precision, recall = precision_and_recall(testing_targets, prediction)
    formula_1 = f1_measure(testing_targets, prediction)

    # Order of these returns must be maintained
    return confuscious_matrix, accuraC, precision, recall, formula_1
