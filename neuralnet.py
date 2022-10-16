import numpy as np
import torch
from torch import tensor
from torch.nn import Sigmoid
import matplotlib.pyplot as plt

class NNLearner:
    def __init__(NN,df,yName,learningRate,numIterations=20,lossFn='MSE',geometry=[10,10]):
        """
        Parameters
        ----------
        df : Pandas dataframe
        yName : String
                Name of the dependent column in the dataframe.
        learningRate : Float
        numIterations : Int
                        Number of iterations for gradient descent.
        lossFn : String
                 Name of the loss function to be used, as specified in dictionary.
        geometry : list of Ints
                   Size and number of hidden layers of the network.
        """
        NN.X,NN.Y = NN.tensorFromDf(df,yName)
        NN.learningRate = learningRate
        NN.numIterations = numIterations
        NN.geometry = [df.shape[1]-1] + geometry + [1]
        NN.lossFunctionDict = { 'MSE' : lambda Ypred,Y : ((Ypred-Y)**2).mean() }
        NN.loss = NN.lossFunctionDict[lossFn]
        NN.coeffs = NN.initCoeffs()
        NN.losses = []
    
    def tensorFromDf(NN,df,*names):
        """
        Parameters
        ----------
        df : Pandas dataframe
        names : String

        Turn a Pandas dataframe into PyTorch tensors. The
        second argument names is optional. When passed, it
        contains the name of the dependent column and it is
        assumed that df contains at least two columns. Then
        we return the independent and dependent columns as
        tensors. If no name is passed, we just turn X into
        a tensor, ensuring it has correct dimensions.
        """
        numRows = df.shape[0]
        for yName in names:
            X = tensor(df.drop(yName,axis=1).values)
            Y = tensor(df[yName].values).reshape(numRows,1)
            return X,Y
        if len(df.shape) == 1:
            return tensor(df.values).reshape(numRows,1)
        return tensor(df.values)

    def initCoeffs(NN):
        """
        Initialise coefficient tensors. This happens
        during initialisation of the class.

        We use Xavier initialisation (since we use a
        Sigmoid activation function).
        """
        dims = [(NN.geometry[i]+1,NN.geometry[i+1]) for i in range(len(NN.geometry)-1)]
        coeffs = [tensor((np.random.rand(*d)-0.5)*np.sqrt(1/d[0])) for d in dims]
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
        X : PyTorch Tensor
            X contains the independent variables
            where each row represents a sample.
            Therefore if X is only one data point,
            it has to have shape (1,numFeatures).
             
        This function computes the forward propagation through
        the neural network.
        """
        res = NN.X
        for c in NN.coeffs:
            res = Sigmoid()(NN.addPadding(res)@c)
        return res
    
    def predictDf(NN,X):
        """
        Parameters
        ----------
        X : Pandas Dataframe
            X contains the independent variables
            where each row represents a sample.
            Therefore if X is only one data point,
            it has to have shape (1,numFeatures).
             
        This function computes the forward propagation through
        the neural network.
        """
        return NN.predict(NN.tensorFromDf(X))

    def learn(NN):
        """
        This function learns coefficients to fit the data
        by repeating numIterations gradient descent steps.
        """
        for i in range(NN.numIterations):
            Ypred = NN.predict(NN.X)
            NN.losses.append(NN.loss(Ypred,NN.Y).item())
            NN.loss(Ypred,NN.Y).backward()
            with torch.no_grad():
                for c in NN.coeffs:
                    c.sub_(NN.learningRate*c.grad)
                    c.grad.zero_()
    
    def trainingAccuracy(NN):
        """
        Computes the training accuracy, i.e. the quotient
        (correctly predicted) / (total number of samples).
        """
        return ((NN.predict(NN.X)>0.5)==NN.Y).sum()/(NN.X.size()[0])
    
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
