import numpy as np

def mse(y_true, y_pred): #available in sklearn.metrics
    """
    Returns mean-squared-error of y_true and y_pred.
    """
    squares = np.power(y_true - y_pred, 2)
    return np.mean(squares)
    
def logloss(y_true, prob_pred): #available in sklearn.metrics
    """
    Returns log-loss of y_true based on predicted probabilities prob_pred.
    """
    terms = y_true * np.log(prob_pred) + (1 - y_true) * np.log(1 - prob_pred)
    return -np.mean(terms)

def rmsle(y_true, y_pred):
    diffs = np.log(y_true + 1) - np.log(y_pred + 1)
    squares = np.power(diffs, 2)
    return np.sqrt(np.mean(squares))
    

def cv_score_minimizing(fold_metrics):
    """
    Returns mean + std of fold_metrics.
    Useful when trying to minimize a loss metric (e.g. minimize MSE).
    """
    return np.mean(fold_metrics) + np.std(fold_metrics)
    
def cv_score_maximizing(fold_metrics):
    """
    Returns mean - std of fold_metrics.
    Useful when trying to maximize a metric (e.g. maximizing area under ROC curve).
    """
    return np.mean(fold_metrics) - np.std(fold_metrics)