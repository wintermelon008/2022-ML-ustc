# XGBoost
import numpy as np
from . import RegTree as M_REG


class XGBoost(object):
    
    def __init__(self, X_train:np.ndarray, y_train:np.ndarray) -> None:
        self.X = X_train
        self.y = y_train
        self.m, self.n = self.X.shape
        self.treeset = []
        self.train_err = []

    def fit(self, T = 10, min_train_err = 1e-6, tree_parameters: dict = {}):
        m, n = self.y.shape
        self.T = T
        y_t = np.zeros((m, 1))
        
        boost_if_silent = 1
        if tree_parameters.__contains__("if_silent"):
            boost_if_silent = tree_parameters["if_silent"]
        
        for i in range(0, self.T):
            tree = M_REG.RegressionTree(self.X, self.y, y_t)
            tree.fit(tree_parameters)
            
            y_t = y_t + tree.predict(self.X)
            self.treeset.append(tree)
            err = tree._get_err(y_t)
            self.train_err.append(err)
            if err < min_train_err:
                break
        
            if boost_if_silent == 0:
                print("Tree No.{} is done.".format(i))
    
    
    def predict(self, X_test:np.ndarray):
        m, n = X_test.shape
        pre = np.zeros((m, 1))
        for tree in self.treeset:
            pre = pre + tree.predict(X_test)
        return pre
    
    
    def RMSE(self, pre:np.ndarray, val:np.ndarray):
        y = val - pre
        m, _ = y.shape
        return np.sqrt((y.T @ y) / m)[0][0]
    
    def R2(self, pre:np.ndarray, val:np.ndarray):
        var = np.var(val)
        return(1 - (self.RMSE(pre, val) ** 2)/ var)