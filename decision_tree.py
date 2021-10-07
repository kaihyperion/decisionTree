import numpy as np

class Node():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
  
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        Output:
            VOID: It should update self.tree with a built decision tree.
        """
        # Uses the ID3() to build the tree
        self._check_input(features)


        self.tree = self.ID3(features, targets, self.attribute_names, default=None)

        return

    def predict(self, features):
        # Looks at a tree and tries to build a list of predicitions

        self._check_input(features)

        predictions = []
        for row in features:
            tree = self.tree
            while tree.attribute_name != "root":  # at a leaf_node: #psuedocode: Not at a leaf node, we want to keep iterating through choosing the best attribute
                if row[tree.attribute_index] == 1:  # If best.attribute_index == 1,
                    tree = tree.branches[1]  # whatever this branch it corresponds to. keep iterating until we get to the bottom(base case)
                else:
                    tree = tree.branches[0]
            predictions.append(tree.value)  # We are iterating through our tree so append. We want to append the value
        return predictions

    def _visualize_helper(self, tree, level):

        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):

        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)
    def ID3(self, features, target, attributes, default=None):
        """
        examples ==> features (np.array): numpy array of size NxF containing features, where N is
            number of examples and F is number of features.
            number of example X = {x_1,...x_n} which is basically features.shape[0]
            each example is a n-tuple of attribute values x_1 = <a_1, ... a_k> if there is k attributes

        attributes ==> targets (np.array): numpy array containing class labels for each of the N
            examples.
        val ==> attribute_names (list): list of strings containing the attribute names for
            each feature (e.g. chocolatey, good_grades, etc.)
        Default

        """

        # if features/examples are empty. If the vertical size of the feature array is \le 1
        if features.size == 0:
            # calculate the most common (0 or 1)
            return default

        # if same classification
        elif len(np.unique(target)) <= 1:
            return np.unique(attributes)[0]

        # if target is empty then return mode(examples)
        elif not attributes:
            mode = statistics.mode(target)
            return mode

        else:
            """
            best <= CHOOSE-ATTRibute(attributes, examples)
            id3(features, attribute_index, targets)
            - We need to cycle thru attribute index
            attribute_indx = for attribute in range(features.shape[0]):
            
            """
            entropy_values = None
            for index in range(features.shape[1]):
                entropy_values = [information_gain(features, index, attributes)]
            best_index = np.argmax(entropy_values)
            best = attributes[best_index]  # String

            # Create tree??
            # Tree <= a new decision tree with root  best
            node = Tree(attribute_name=best, attribute_index=float(best_index), branches=[])

            # Lets make a new features
            new_attributes = [i for i in self.attribute_names if i != best]
            # Splitting into branches => 0 or 1
            best_column = features[:, best_index]
            for branch in np.unique(best_column):  # list of 0 and 1
                new_features = features[np.where(best_column == branch)]
                new_target = target[np.where(best_column == branch)]
                sub_tree = self.ID3(new_features, new_target, new_attributes, None)
                node.branches.append(sub_tree)
                #
                # np.delete(attributes, best_column != branch) #Delete everything except branch
                # del val[best_index]
                # final_feature = np.delete(np.delete(features, best_column != branch, 0), best_index, 1)
                # sub_tree = ID3(self, final_feature, attributes, val, default = None)
                # node[best][branch] = sub_tre
            return node
            
# def prior_probability(column):
#     examples, counts = np.unique(column, return_counts=True)  # this will sort it by unique 0 and 1 and return the counts of each.
#     # examples = array([0,1])    counts = array([#of 0's ,# of 1])
#     negative = examples[0]
#     for i in range(len(examples)):
#
#     negative_pp, positive_pp = (counts[:] / sum(counts))
#     negative_pp =
   # return negative_pp, positive_pp


def entropy(column):
    #nn, pp = prior_probability(column)
    # Entropy calculation:
    Entropy_Value = 0
    elements, counts = np.unique(column, return_counts=True)
    Entropy_Value = -np.sum([
        for i in range(len(elements)):
            ((counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)))
    return Entropy_Value            

def information_gain(features, attribute_index, targets):
    # attribute_index is basically like the split point where or Looking at each indiivudal attributes
    # Calculating ENTIRE-DATA-SET Entropy
    entropy_S = entropy(targets)  # This is important for the "Before Split"

    # We have to basically recurse the entropy for EACH VALUE OF THAT attribute
    # Think about the Outlook attribute -> sunny, rainy, overcast  VALUES
    # Gather each and one of entropy for all that VALUES
    # Find the Average information entropy_S
    # Calculate and return Gain Value
    #column = features[:, attribute_index]

    entropy_index = entropy(features[:, attribute_index])  # This is important for "After-Split"


    # Since this a binary split between 0 or 1...
    # WE just have to get the Gain = entropy_S - attribute_index_entropy

    return entropy_S - entropy_index

if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
