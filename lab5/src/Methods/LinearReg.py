# 线性回归模型
import numpy as np



class LinearRegression:

    def __init__(self):
        pass


    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        
    def predict(self, X:np.ndarray) -> np.ndarray: 
        y_pre = X @ (np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y)
        y_pre = y_pre.reshape(-1, 1)
       
        return y_pre
