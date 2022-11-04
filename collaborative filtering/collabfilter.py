import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm # progress bar

class collabFilter(torch.nn.Module):
    def __init__(self, num_ids, num_features):
        """
        Parameters
        ----------
        num_ids : List[int]
                  List with two entries, the number of unique IDs
                  in the first resp. second column (in other words
                  the number of unique customer and item IDs).
        num_rows : int
                   Number of rows in the dataframe (i.e. the number
                   of non-zero entries in the matrix representation).
        num_features : int
                       Number of features each ID has.
        """
        super().__init__()
        variance = 0.01
        self.factors0 = torch.nn.Parameter(variance * torch.randn(num_ids[0],num_features))
        self.factors1 = torch.nn.Parameter(variance * torch.randn(num_ids[1],num_features))
        self.bias0 = torch.nn.Parameter(variance * torch.randn(num_ids[0],1))
        self.bias1 = torch.nn.Parameter(variance * torch.randn(num_ids[1],1))
        self.sigmoid_modified = lambda x : 1.2 * torch.nn.Sigmoid()(x) - 0.1

    def forward(self, X):
        """
        We assume that the predictions are roughly in the range [0,1].
        """
        num_rows = X.size()[0]
        res = (self.factors0[X[:,0]] * self.factors1[X[:,1]]).sum(axis=1).reshape(num_rows,1)
        res += self.bias0[[X[:,0]]] + self.bias1[[X[:,1]]]
        res = self.sigmoid_modified(res)
        return res

class collabfilter_learner:
    def __init__(CF, df, num_features = 10, num_iterations = 100, learning_rate = 1e-4, ratio = 0.2):
        """
        Parameters
        ----------
        df : Pandas dataframe
        num_features : int
        num_iterations : int
        learning_rate : float
        ratio : float
                The ratio of training to validation
                split. A ratio of 0.2 means that 
                around 20% of the data are used for
                validation.

        We assume the input df to be of the form
           ID1 ID2 rating
        0  ... ... ...
        1  ... ... ...
        2  ... ... ...
        ...
        IDs may be ints, strings, etc.
        Ratings are assumed to be in the range [0,1].

        For predictions, we assume the dataframe to
        be of the form
           ID1 ID2
        0  ... ...
        1  ... ...
        2  ... ...
        ...
        (in particular we require the column names to
        be identical!)
        """
        CF.ID_column_names = list(df.columns)[:2]
        CF.rating = list(df.columns)[-1]
        CF.num_iterations = num_iterations
        CF.num_features = num_features
        CF.learning_rate = learning_rate
        # Dictionaries to convert the IDs in dataframe to indices
        CF.lookups = {c : dict(zip(df[c].unique(),range(len(df[c].unique())))) for c in CF.ID_column_names}
        # Number of distinct IDs in each column
        CF.num_ids = [len(CF.lookups[key]) for key in CF.lookups]
        # Two lists to contain the loss at each iteration for training
        # and validation loss
        CF.losses_train = []
        CF.losses_val   = []
        # Transforming the input ID columns into indexed Torch tensors
        # and splitting into training and validation sets.
        CF.X_train, CF.X_val, CF.Y_train, CF.Y_val = CF.train_val_split(df,ratio)
        # Number of ratings
        CF.num_rows = CF.X_train.size()[0]
        CF.num_rows_val = CF.X_val.size()[0]
        # Storing the whole dataframe for convenience
        CF.df = df
        # Initialising the model
        CF.model = collabFilter(CF.num_ids,CF.num_features)
    
    def df_to_indexed_tensor(CF,df):
        inds = [df[c].map(CF.lookups[c]) for c in CF.ID_column_names]
        tensors = [torch.Tensor(x).long().reshape(df.shape[0],1) for x in inds]
        return torch.cat(tensors,axis=1)

    def train_val_split(CF,df,ratio):
        X = CF.df_to_indexed_tensor(df)
        Y = torch.Tensor(df[CF.rating]).reshape(X.size()[0], 1)
        inds = torch.rand(X.size()[0]) < ratio
        return X[~inds], X[inds], Y[~inds], Y[inds]

    def learn(CF):
        # Setting up loss function and optimizer
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(CF.model.parameters(), lr=CF.learning_rate)
        # Iterate
        for i in tqdm(range(CF.num_iterations)):
            # Compute predictions and loss on training set
            Y_pred = CF.model(CF.X_train)
            loss_train = criterion(Y_pred, CF.Y_train)
            CF.losses_train.append(loss_train.item()/CF.num_rows)
            # Take an optimiser step
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            # Compute the validaton loss
            loss_val = criterion(CF.model(CF.X_val),CF.Y_val)
            CF.losses_val.append(loss_val.item()/CF.num_rows_val)

    def predict(CF, df):
        """
        Parameters
        ----------
        df : Pandas dataframe
             The data to compute predictions on. This is assumed
             to be of the same exact form as the training dataframe
             except that we do not expect a rating column to be
             present.
        """
        X = CF.df_to_indexed_tensor(df)
        return CF.model(X)

    def plot_losses(CF):
        """
        Creates a plot of the loss function against the
        number of iterations.
        """
        fig = plt.figure()
        ax = plt.subplot()
        ax.plot(CF.losses_train, label = 'Training loss')
        ax.plot(CF.losses_val, label = 'Validation loss')
        plt.xlabel('Number of iterations')
        plt.ylabel('Loss')
        plt.title('Loss against # of iterations')
        ax.legend()
        plt.show()  

    def highest_bias(CF, column_name, n = 10):
        """
        Parameters
        ----------
        column_name : String
                      Name of the column in the dataframe
                      for which the highest biases should
                      be computed.
        n : Int
        
        For both columns, returns the n IDs (entries in the column)
        with the highest biases.
        """
        idx = CF.ID_column_names.index(column_name)
        inds0 = CF.model.bias0.sort(dim=0, descending=True).indices[:n]
        inds1 = CF.model.bias1.sort(dim=0, descending=True).indices[:n]
        inds = [inds0,inds1][idx]
        return list(CF.df[CF.ID_column_names[1]].unique()[inds].flatten())