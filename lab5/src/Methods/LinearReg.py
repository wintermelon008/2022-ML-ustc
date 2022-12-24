# 线性回归模型
import numpy as np
import pickle
import time


class LinearRegression:

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self._X = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        

    def fit(self):
        pass
        
    def predict(self, X:np.ndarray) -> np.ndarray: 
        y_pre = X @ self._X
        y_pre = y_pre.reshape(-1, 1)
       
        return y_pre
    
    
    def save(self, filename = None):
        
        if filename == None:
            t_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
            filename = "./model/Model_LinearReg" + "_" + t_str + ".dat"
            
        pickle.dump(self, open(filename, "wb"))


                