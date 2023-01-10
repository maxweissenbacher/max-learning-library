import pandas as pd
import numpy as np
from statistics import mode
from random import sample
import abc

class tree:
    def __init__(self):
        """
        A simple binary tree class that also holds
        all the information we need for a decision
        tree in each node. All parameters default
        to None.
        
        Explanation of parameters:
        left and right are used to store child nodes
        of the given node (if they exist, they will
        be of type tree again, if they do not exist,
        this means that the node is terminal, i.e. a
        leaf node).
        
        feature and threshold are used to remember
        the optimal feature and threshold to split
        at this node. We use the convention that if
        x[feature] <= threshold, then we go to the
        left child node, otherwise we go right.
        
        y is used to store the class of the current
        node (for a classification tree) or the
        expected value of the node (for a regression
        tree).
        
        score stores the score of the current node
        (Gini score for a classification tree and
        mean squared error for a regression tree).
        """
        self.left  = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.y = None
        self.score = None

    def __eq__(self,other):
        if self is None and other is None:
            return True
        if self is None or other is None:
            return False
        if not isinstance(other,tree):
            raise NotImplementedError
        if self.feature != other.feature:
            return False
        if self.threshold != other.threshold:
            return False
        if self.y != other.y:
            return False
        if self.score != other.score:
            return False
        if self.left == other.left and self.right == other.right:
            return True
        return False
    
    def depth(self, d = 1):
        """
        Return the depth of the tree. We define a single node
        with no children to have depth 1. A tree of depth 2
        is a 'stump', etc.

        Args:
            d (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        if self.left is None and self.right is None:
            return d
        return max(self.left.depth(d+1),self.right.depth(d+1))
        
class decisionTreeLearner(abc.ABC):
    def __init__(self, X, Y, max_depth, min_elements = None, feature_fraction = 1.0):
        """
        This is an abstract class that has both a
        classifier and a regressor implementation
        as subclasses. This class can not be
        instantiated on its own, since the score
        method, value method the find_best_split method
        must be defined in the respective subclass.

        Args:
            X (Pandas DataFrame): Holds the features. Features
                                  are assumed to be arranged in
                                  columns, with rows representing
                                  different samples. All entries
                                  are assumed to be numeric.
                                  Therefore categorical data should
                                  be either one-hot encoded or
                                  converted into numerical categories.
            Y (Pandas Series): Holds the classes of each sample.
            max_depth (int): Maximum depth of the decision tree.
            min_elements (int, optional): Minimum number of
                                          elements in each node.
                                          If left unspecified,
                                          we use sqrt(num_samples).
            feature_fraction (float): At each node, we subsample
                                      and only consider a random
                                      proportion of features to
                                      compute the optimal split.
                                      Defaults to 1.0 (which means
                                      no subsampling).
        """
        # Storing data 
        self.X = X
        self.Y = Y
        self.num_features = len(X.columns)
        self.features = list(X.columns)
        # Hyperparameters
        self.max_depth = max_depth
        self.min_elements = int(np.sqrt(len(Y))) if min_elements is None else min_elements
        # Number of features that will be considered during each split
        self.num_subsampling_features = int(feature_fraction * self.num_features)
        if self.num_subsampling_features == 0:
            raise RuntimeError('0 features will be selected for splitting. Increase the feature_fraction.')
        # Initialise an empty decision tree
        self.decisionTree = tree()
        # Once we make a prediction for all data, store it here to avoid unnecessary computation.
        self.predictions = None

    @abc.abstractmethod
    def value(self,Y):
        """
        We need to assign each node a value based
        on the samples contained in Y. The implementation
        need not make a call to super.value().
        """
        return None

    @abc.abstractmethod
    def score(self,Y):
        """
        We need a score function to be implemented by
        the subclass. The implementation need not make
        a call to super.score().
        """
        return None

    @abc.abstractmethod
    def find_best_split(self,X,Y):
        """
        We need a method that finds the best possible
        split based on the score function. The
        implementation is different for the regression
        and classification case, so we leave it to
        the subclass. The implementation need not
        make a call to super.find_best_split().
        """
        return None

    def sample_features(self,X):
        """
        Select a random subset of features in X.
        We select num_subsampling_features many.
        The rows of X remain unchanged.

        Args:
            X (Pandas DataFrame): Features
        """
        if self.num_subsampling_features == self.num_features:
            return X
        return X[sample(self.features,self.num_subsampling_features)]

    def __learn_tree(self,X,Y,depth=1):
        """
        The actual learning takes place here.
        We make recursive calls to this function
        to generate the tree.

        Args:
            X (Pandas DataFrame): Features
            Y (Pandas Series): Categories
            depth (int, optional): Current iteration depth. Defaults to 1.

        Returns:
            tree: Decision tree constructed recursively.
        """
        # Compute a tentative split at this node
        feature, threshold, score_children = self.find_best_split(self.sample_features(X),Y)
        L_indices = X[feature] <= threshold
        # Compute y and score for the node
        decision_tree = tree()
        decision_tree.y = self.value(Y)
        decision_tree.score = self.score(Y)
        # We do not split at this node if max. depth exceeded
        if (depth >= self.max_depth):
            return decision_tree
        # ... or too few elements are in each child node
        if (sum(L_indices) < self.min_elements or sum(~L_indices) < self.min_elements):
            return decision_tree
        # ... or the splitting offers no improvement
        if (not score_children < decision_tree.score):
            return decision_tree
        # Otherwise, split again at the node
        decision_tree.feature = feature
        decision_tree.threshold = threshold
        # Recursively compute the children
        decision_tree.left = self.__learn_tree(X[L_indices],Y[L_indices],depth+1)
        decision_tree.right = self.__learn_tree(X[~L_indices],Y[~L_indices],depth+1)
        return decision_tree
    
    def learn(self):
        """
        This function is called to learn the decision tree.
        The function makes recursive calls to the __learn_tree
        function.

        Returns:
            tree: Learned decision tree
        """
        self.decisionTree = self.__learn_tree(self.X,self.Y)
        
    def __predictRow(self,node,X):
        """
        Compute prediction for a given decision tree
        and one row of data recursively.

        Args:
            node (tree): The decision tree to use
            X (Pandas Series): One row of data
        """
        if node.left is None or node.right is None:
            return node.y
        if X[node.feature] <= node.threshold:
            return self.__predictRow(node.left,X)
        else:
            return self.__predictRow(node.right,X)
    
    def predict(self,X):
        """
        Compute prediction on a given dataframe.

        Args:
            X (Pandas DataFrame or Pandas Series): Features
        
        Returns:
            Pandas Series: Series containing the predictions of the model.
        """
        if self.decisionTree == tree():
            raise RuntimeError('Cannot make predictions before learning the model. Call the method learn() first.')
        if isinstance(X,pd.Series):
            X = X.to_frame().T
        return X.apply(lambda x: self.__predictRow(self.decisionTree,x), axis=1)

class decisionTreeClassifier(decisionTreeLearner):
    def __init__(self,X,Y,max_depth, min_elements = None, feature_fraction = 1.0):
        """
        The classifier implementation of the decision tree.
        Since the algorithm is essentially unchanged compared
        to the regressor case, the only thing we need to
        specify here are the score function (Gini score)
        and the find_best_split method (which computes the
        best possible split based on the Gini score).
        
        Supports multiple categories encoded as multiple
        values in Y.
        
        Arguments are the same as for the parent class.
        """
        super().__init__(X,Y,max_depth, min_elements, feature_fraction) 
    
    def value(self,Y):
        return mode(Y)
    
    def score(self,Y):
        """
        Compute Gini score of a given set of rows.
        Accepts either a dataframe or a dictionary.

        Args:
            Y (DataFrame): Contains the categories assigned
                           to each row. We assume that only
                           the rows of interest are passed
                           to the function.
            Y (dictionary): Keys are categories, values are
                            counts          
        """
        if isinstance(Y,pd.Series):
            if (len(Y) == 0):
                return 1
            probs = np.array([sum(Y==c)/len(Y) for c in Y.unique()])
            return 1-sum(probs**2)
        if isinstance(Y,dict):
            arr = np.array(list(Y.values()))
            if (sum(arr) == 0):
                return 1
            probs = arr / sum(arr)
            return 1-sum(probs**2)
    
    def find_best_split(self,X,Y):
        """
        Find the split (i.e. feature and threshold)
        for a given dataset, using weighted Gini
        score of the two groups as a metric.
        
        Args:
            X (Pandas DataFrame): Features
            Y (Pandas Series): Categories
            
        Returns:
            same datatype as elements of Y : feature used in split
            float : The threshold for the best split
            float : The weighted Gini score of the child nodes
        """
        best_gini = 1.
        num_samples = len(Y)
        threshold = -np.inf
        if len(Y) == 0:
            raise TypeError('Passing an empty vector.')
        
        for feature in X.columns:
            L_categories = {c : 0 for c in Y.unique()}
            R_categories = {c : sum(Y==c) for c in Y.unique()}
            num_left = 0 # Number of elements in left group
            for x,y in sorted(zip(X[feature],Y)):
                R_categories[y] -= 1
                L_categories[y] += 1
                num_left += 1
                if threshold == x:
                    continue
                threshold = x
                L_gini  = self.score(L_categories)
                R_gini  = self.score(R_categories)
                w = num_left/num_samples
                score = w * L_gini + (1-w) * R_gini
                if score < best_gini:
                    best_gini = score
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold, best_gini
    
    @property
    def accuracy(self):
        if self.predictions is None:
            self.predictions = self.predict(self.X)
        return sum(self.predictions == self.Y)/len(self.Y)
        
class decisionTreeRegressor(decisionTreeLearner):
    def __init__(self,X,Y,max_depth, min_elements = None, feature_fraction = 1.0):
        """
        The regressor version of a decision tree. We need
        only implement the value, score and splitting
        functions. Supports only one-dimensional targets.

        Args:
            X (Pandas DataFrame): Features
            Y (Pandas Series): Values to fit
        Arguments otherwise the same as for the parent class.
        """
        super().__init__(X,Y,max_depth, min_elements, feature_fraction)
    
    def value(self,Y):
        """
        Return mean of samples in a node.

        Args:
            Y (Pandas Series): Values to fit
        """
        return Y.mean()
    
    def score(self,Y):
        """
        Return mean squared error of the samples
        in a given node.

        Args:
            Y (Pandas Series): Values to fit
        """
        if len(Y) == 0:
            return 0
        return ((Y-Y.mean())**2).mean()
    
    def find_best_split(self, X, Y):
        """
        Find best split in a given node, using
        mean squarred error as a metric.

        Args:
            X (Pandas DataFrame): Features
            Y (Pandas Series): Values to fit
        """
        if len(Y) == 0:
            raise TypeError('Passing an empty vector.')
        best_score = np.inf
        num_samples = len(Y)
        threshold = -np.inf
        
        for feature in X.columns:
            L_sum   = 0 # Sum of elements in left group
            L_sqsum = 0 # Squared sum of elem in left group
            R_sum   = Y.sum() # Sum of right group
            R_sqsum = (Y**2).sum() # Squared sum of right group
            num_left  = 0 # Number of elements in left group
            num_right = num_samples # Number of elements in right group
            for x,y in sorted(zip(X[feature],Y)):
                L_sum   += y
                L_sqsum += y**2
                R_sum   -= y
                R_sqsum -= y**2
                num_left  += 1
                num_right -= 1
                if threshold == x or num_right == 0:
                    continue
                threshold = x
                L_var  = L_sqsum/num_left - (L_sum/num_left)**2
                R_var  = R_sqsum/num_right - (R_sum/num_right)**2
                w = num_left/num_samples
                total_var = w * L_var + (1-w) * R_var
                if total_var < best_score:
                    best_score = total_var
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold, best_score
    
    @property
    def mse(self):
        """
        Compute mean squared error of the fitted model.
        """
        if self.decisionTree == tree():
            raise RuntimeError('Cannot compute accuracy before learning model. Call the method learn() first.')
        return ((self.predict(self.X)-self.Y)**2).mean()
    
    @property
    def abs(self):
        """
        Compute absolute error of the fitted model.
        """
        if self.decisionTree == tree():
            raise RuntimeError('Cannot compute accuracy before learning model. Call the method learn() first.')
        return (abs(self.predict(self.X)-self.Y)).mean()