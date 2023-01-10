import pandas as pd
from tqdm import tqdm # progress bar
import matplotlib.pyplot as plt
from decisiontree import decisionTreeRegressor

class gradientBoostedRegressionTree:
    def __init__(self, X, Y, num_models, learning_rate = 0.1, tree_depth = 2):
        # Data
        self.X = X
        self.Y = Y
        # Hyperparameters
        self.num_models = num_models
        self.learning_rate = learning_rate
        self.tree_depth = tree_depth
        # Store the sequence of models
        self.models = []
        # Store the MSE at each iteration
        self.mse_history = []
        
    def learn(self, show_progress_bar = True):
        # Initial prediction is the mean
        Y_predicted = pd.Series(self.Y.mean(),self.Y.index)
        # Error based on L^2 (MSE) loss function
        error = self.Y - Y_predicted
        self.mse_history.append((error**2).mean())
        for i in tqdm(range(self.num_models), disable = not show_progress_bar):
            learner = decisionTreeRegressor(self.X,error,max_depth=self.tree_depth)
            learner.learn()
            if (learner.decisionTree.depth() == 1):
                print(f'Further splitting does not lead to improvement in MSE score. Stopping early at iteration {i}.')
                break
            self.models.append(learner)
            Y_predicted += self.learning_rate*learner.predict(self.X)
            error = self.Y - Y_predicted
            self.mse_history.append((error**2).mean())
            
    def predict(self,X):
        res = self.Y.mean()
        if self.models == []:
            raise RuntimeError('Cannot predict before training the model. Call learn() first.')
        for model in self.models:
            res += self.learning_rate*model.predict(X)
        return res
    
    @property
    def mse(self):
        if not self.mse_history:
            raise RuntimeError('Cannot compute MSE before training the model. Call learn() first.')
        return self.mse_history[-1]
    
    def plot_mse_history(self):
        plt.plot(self.mse_history)
        plt.xlabel('Number of models')
        plt.ylabel('MSE')
        plt.title('MSE against # of models')
        plt.show()