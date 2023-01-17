import numpy as np
import pandas as pd
import torch
from itertools import permutations
from tqdm import tqdm

class kmeans:
    def __init__(self, df, k, metric = None):
        """
        Parameters
        ----------
        df : Pandas Dataframe
        k : Int
        metric : String (or None)

        Implementation of k-means algorithm, for data
        stored in the dataframe df. We assume that rows
        represent separate points and columns represent
        the coordinates of each point. Labels will be
        indices 0,...,k-1.
        """
        self.X = torch.Tensor(df.values)
        self.k = k
        self.metric = metric
        # Initialise centers by randomly selecting k distinct data points
        self.centers = self.X[np.random.choice(df.shape[0],k,replace=False)]
        # Initialise labels to all 0 before learning
        self.labels = torch.zeros(df.shape[0])
    
    def closest_center(self, X):
        """
        Parameters
        ----------
        X : Torch Tensor

        Returns a list of indices, where each index is
        the the center point closest to the corresponding
        row in X, as measured by the metric specified in
        the metric argument. Default is Euclidean.
        """
        # Compute distances to each center
        distances = [torch.linalg.norm(X - c, axis=1, ord = self.metric) for c in self.centers]
        return np.argmin(torch.stack(distances), axis=0)
    
    def learn(self):
        # Iterate until the centers do not move anymore
        diff = self.centers
        def generator():
            while sum(list(map(sum,diff))) != 0:
                yield
        for _ in tqdm(generator()):
            self.labels = self.closest_center(self.X)
            # Update the centers by taking the mean of every cloud
            updated_centers = [self.X[self.labels == i].mean(axis=0) for i in range(self.k)]
            # Compute the differences between old and new centers
            diff = [i-j for (i,j) in zip(updated_centers,self.centers)]
            # Store the updated centers in the centers variable
            self.centers = updated_centers
            
    def predict(self,df):
        """
        Parameters
        ----------
        df : Pandas Dataframe

        Return predicted labels for a given dataframe.
        Again we assume that points are stored in rows,
        where the columns represent the coordinates of
        a point.
        """
        X = torch.Tensor(df.values)
        return pd.Series(self.closest_center(X))

    def accuracy(self, Y, show_progress_bar = False):
        """
        Parameters
        ----------
        Y : Pandas Series
            We assume that Y only contains values 0,...,k-1.
        
        Computes the accuracy of our computed labels to the given
        labels Y (assuming that the labels Y are known to be correct).
        We take into account the fact that our labels might be a 
        permutation of the labels as given in Y. We return both the
        accuracy and the permutation necessary to relate the labels in
        Y to the computed labels.
        """
        perm = permutations(range(self.k))
        min_error = np.inf
        min_p = None
        # Convert labels from Tensor to pandas.Series
        labels_pd = pd.Series(self.labels)
        # For each permutation, find the rate of misclassified points.
        for p in tqdm(perm, disable = not show_progress_bar):
            d = dict(enumerate(p))
            error = (Y.map(d)-labels_pd != 0).mean()
            if error < min_error:
                min_error = error
                min_p = p
        return min_error,min_p
