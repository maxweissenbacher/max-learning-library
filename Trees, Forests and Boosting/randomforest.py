import pandas as pd
from random import choices
from tqdm import tqdm # progress bar

from decisiontree import decisionTreeClassifier

class randomForestClassifier:
    def __init__(self, 
                 X, 
                 Y, 
                 num_trees, 
                 max_depth, 
                 feature_fraction, 
                 observation_fraction=1.0):
        self.X = X
        self.Y = Y
        if not all(X.index == Y.index):
            raise RuntimeError('Indexation of rows of X and Y does not agree.')
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.feature_fraction = feature_fraction
        self.num_observed_samples = int(observation_fraction * len(Y))
        if self.num_observed_samples == 0:
            raise RuntimeError('0 samples have been selected for each tree. Increase observation_fraction.')
        # Store the trained models like so
        self.models = []
        # Once we make a prediction for all data, store it here to avoid unnecessary computation.
        self.predictions = None
        # Similarly, once we compute the probability of predicting each class for all data, we store it.
        self.probabilities = None
    
    def learn(self):
        for _ in tqdm(range(self.num_trees)):
            indices = choices(list(self.X.index),k=self.num_observed_samples)
            learner = decisionTreeClassifier(
                X = self.X.loc[indices],
                Y = self.Y[indices],
                max_depth = self.max_depth,
                feature_fraction = self.feature_fraction
            )
            learner.learn()
            self.models.append(learner)
            
    def predict(self, X):
        if not self.models:
            raise RuntimeError('Cannot predict before training the model. Call learn() method first.')
        preds = pd.DataFrame([model.predict(X) for model in self.models])
        # If there are several modes, we return only one of them for simplicity.
        return preds.mode(axis=0).loc[0]
    
    def probs(self, X):
        if not self.models:
            raise RuntimeError('Cannot predict before training the model. Call learn() method first.')
        preds = pd.DataFrame([model.predict(X) for model in self.models])
        res = pd.DataFrame()
        for y in self.Y.unique():
            res[y] = preds[preds == y].count(axis=0)/self.num_trees
        return res
    
    @property
    def accuracy(self):
        if self.predictions is None:
            self.predictions = self.predict(self.X)
        return sum(self.predictions == self.Y)/len(self.Y)
            