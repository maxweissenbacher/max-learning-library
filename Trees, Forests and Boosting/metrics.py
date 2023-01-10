import numpy as np
import matplotlib.pyplot as plt

def roc(learner,X,Y,cat):
    # Compute the probability distribution of each category    
    probs = learner.probs(X)
    # Set up empty arrays to contain the points
    ROC_points_x = np.array([])
    ROC_points_y = np.array([])
    # For the following values of t, compute true
    # positive and false positive ratios.
    t_discretisation = np.linspace(0,1,100)
    for t in t_discretisation:
        num_cat  = (Y == cat).sum()
        num_rest = len(Y) - num_cat
        # True positives and false positives
        tp = sum((probs[cat] > t)[Y == cat])
        fp = sum((probs[cat] > t)[Y != cat])
        # True positive ratio and false positive ratio
        tpr = tp/num_cat
        fpr = fp/num_rest
        # Store for plotting
        ROC_points_x = np.append(ROC_points_x, fpr)
        ROC_points_y = np.append(ROC_points_y, tpr)
    ROC_points_x = np.append(ROC_points_x, 1.)
    ROC_points_y = np.append(ROC_points_y, 1.)
    # Plot results
    plt.scatter(ROC_points_x,ROC_points_y, color = 'green')
    plt.plot(ROC_points_x,ROC_points_y, color = 'tab:green')
    plt.plot(t_discretisation,t_discretisation, color = 'maroon')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC')
    plt.show()
    
def precision(learner,X,Y):
    #if learner.predictions is None:
    #    learner.predictions = learner.predict(X)
    preds = learner.predict(X)
    precision_dict = {}
    for category in learner.Y.unique():
        num_preds = sum(preds == category)
        num_true_pos = sum(Y[preds == category] == category)
        precision_dict[category] = num_true_pos/num_preds
    return precision_dict
    
def recall(learner,X,Y):
    #if self.predictions is None:
    #    self.predictions = self.predict(self.X)
    preds = learner.predict(X)
    recall_dict = {}
    for category in learner.Y.unique():
        num_cat = sum(Y == category)
        num_true_pos = sum(preds[Y == category] == category)
        recall_dict[category] = num_true_pos/num_cat
    return recall_dict
    
def F1(learner,X,Y):
    prec = precision(learner,X,Y)
    rec = recall(learner,X,Y)
    F1_dict = {}
    if prec.keys() != rec.keys():
        raise RuntimeError('Constructed the keys for precision and recall incorrectly.')
    for key in prec.keys():
        p = prec[key]
        r = rec[key]
        F1_dict[key] = 2*p*r/(p+r)
    return F1_dict