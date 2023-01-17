import numpy as np
from numpy.random import uniform
import pandas as pd
import torch
from torch import tensor
from torch.nn import Sigmoid, Softmax, ReLU
import matplotlib.pyplot as plt
from tqdm import tqdm # progress bar

class NNLearner:
    def __init__(NN,
                 X,
                 Y,
                 learningRate,
                 numIterations=20,
                 lossFn='entropy',
                 geometry=[],
                 dropoutRate = 0.0):
        """
        Parameters
        ----------
        X : Pandas DataFrame or Series
        Y : Pandas DataFrame or Series
            We assume that categories are one-hot encoded.
        learningRate : Float
        numIterations : Int
                        Number of iterations for gradient descent.
        lossFn : String
                 Name of the loss function to be used, as specified in dictionary.
        geometry : list of Ints
                   Size and number of hidden layers of the network.
                   Default is no hidden layers (= logistic regression).
        dropoutRate : Float
                      The percentage of hidden nodes to be deactivated
                      for dropout regularisation while learning.
        """
        NN.X,NN.Y = NN.convert_to_tensor(X), NN.convert_to_tensor(Y)
        NN.Y_columns = Y.columns if isinstance(Y,pd.DataFrame) else None
        NN.learningRate = learningRate
        NN.numIterations = numIterations
        NN.dropoutRate = dropoutRate
        NN.geometry = [NN.X.shape[1]] + geometry + [NN.Y.shape[1]]
        # Choose the desired loss function
        if lossFn == 'entropy':
            # If data are one hot encoded, use this formula:
            if NN.Y.shape[1] == 1:
                NN.loss = lambda Ypred,Y : -(Y * torch.log(Ypred) + (1-Y) * torch.log(1-Ypred)).mean()
            # In the case of binary classification (no one hot encoding:
            else:
                NN.loss = lambda Ypred,Y : -(torch.log(Ypred) * Y).sum()/Y.shape[0]
        if lossFn == 'MSE':
            NN.loss = lambda Ypred,Y : ((Ypred-Y)**2).mean()
        NN.coeffs = NN.initCoeffs()
        NN.losses = []
    
    def __repr__(NN):
        return 'Simple implementation of a neural network with specifiable geometry and learning rate. The network uses a Sigmoid activation function throughout and makes use of Xavier initialisation.'

    def convert_to_tensor(NN,X):
        """
        Parameters
        ----------
        X : Pandas DataFrame, Pandas Series or PyTorch Tensor
        names : String

        Type sensitive: if input is a Torch Tensor, do nothing.
        If input is a Pandas DataFrame or Series, we convert
        it into a Tensor.
        
        The second argument names is optional. When passed, it
        contains the name of the dependent column and it is
        assumed that X contains at least two columns. Then
        we return the independent and dependent columns as
        tensors. If no name is passed, we just turn X into
        a tensor, ensuring it has correct dimensions.
        """
        # If X is a Tensor, just return X
        if isinstance(X,torch.Tensor):
            return X
        numRows = X.shape[0]
        # If X is a Series, reshape into a column vector
        if len(X.shape) == 1:
            return tensor(X.values,dtype=float).reshape(numRows,1)
        # Otherwise, turn X into a Tensor and return
        return tensor(X.values,dtype=float)

    def initCoeffs(NN):
        """
        Initialise coefficient tensors. This happens
        during initialisation of the class.

        We initialise weights with mean 0 and variance
        of layer L to be 2/n_L, where n_L is the number
        of input nodes in layer L.
        """
        dims = [(NN.geometry[i]+1,NN.geometry[i+1]) for i in range(len(NN.geometry)-1)]
        coeffs = [tensor((np.random.rand(*d)-0.5)*np.sqrt(2/d[0])) for d in dims]
        for c in coeffs:
            c.requires_grad_()
        return coeffs
    
    def addPadding(NN,X):
        """
        Parameters
        ----------
        X : PyTorch tensor
    
        Add a column of ones to X.
        """
        return torch.nn.ConstantPad1d((0,1), 1.)(X)
    
    def predict(NN,X):
        """
        Parameters
        ----------
        X : PyTorch Tensor or Pandas DataFrame
            X contains the independent variables
            where each row represents a sample.
            Therefore if X is only one data point,
            it has to have shape (1,numFeatures).
             
        This function computes the forward propagation through
        the neural network. Returns the same datatype that is
        input (i.e. if X is a Dataframe, we return a dataframe,
        else we return a PyTorch Tensor).
        """
        # Remember input data type
        input_is_tensor = isinstance(X,torch.Tensor)
        # Forward propagation through hidden layers
        res = NN.convert_to_tensor(X) # make sure X is Tensor
        for i in range(len(NN.coeffs)-1):
            res = NN.addPadding(res)@NN.coeffs[i]
            res = ReLU()(res) # ReLU activation function
        # On the last layer of the network, map through Sigmoid
        res = NN.addPadding(res)@NN.coeffs[-1]
        res = Sigmoid()(res)
        # For a multiclass problem, apply softmax.
        if NN.Y.shape[1] > 1:
            res = Softmax(dim=1)(res)
        # Return the correct data type
        if input_is_tensor:
            return res
        elif NN.Y_columns is not None:
            # Convert to DataFrame
            res = pd.DataFrame(res.detach().numpy())
            res.columns = NN.Y_columns
            res.index = X.index
            return res
        else:
            res = pd.Series(res.detach().numpy().flatten())
            res.index = X.index
            return res

    def predictCat(NN,X):
        if isinstance(X,pd.DataFrame) or isinstance(X,pd.Series):
            return NN.predict(X).idxmax(axis=1)
        if isinstance(X,torch.Tensor):
            return NN.predict(X).argmax(axis=1)

    def learn(NN):
        """
        This function learns coefficients to fit the data
        by repeating numIterations gradient descent steps.
        """
        for _ in tqdm(range(NN.numIterations)):
            Ypred = NN.predict(NN.X)
            NN.losses.append(NN.loss(Ypred,NN.Y).item())
            NN.loss(Ypred,NN.Y).backward()
            with torch.no_grad():
                for c in NN.coeffs:
                    c.sub_(NN.learningRate*c.grad)
                    c.grad.zero_()
                    dropout = torch.Tensor(uniform(size=c.size()) > NN.dropoutRate)
                    c = torch.mul(c,dropout)
    
    @property
    def trainingAccuracy(NN):
        """
        Computes the training accuracy, i.e. the quotient
        (correctly predicted) / (total number of samples).
        """
        if NN.Y.shape[1] == 1:
            preds = NN.predict(NN.X)>0.5
            return float((preds == NN.Y).mean(dtype=float))
        else:
            preds  = NN.predictCat(NN.X)
            actual = NN.Y.argmax(axis=1)
            return float((preds == actual).mean(dtype=float))
    
    def plotLosses(NN):
        """
        Creates a plot of the loss function against the
        number of iterations.
        """
        plt.plot(NN.losses)
        plt.xlabel('Number of iterations')
        plt.ylabel('Loss')
        plt.title('Loss against # of iterations')
        plt.show()    
