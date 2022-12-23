# 线性回归模型
import numpy as np


def Labelize(pre: np.ndarray) -> list:
    m, _ = pre.shape
    result = []
    
    for i in range(m):
        if pre[i][0] >= 2.5:
            result.append(3)
        elif pre[i][0] >= 1.5:
            result.append(2)
        elif pre[i][0] >= 0.5:
            result.append(1)
        else:
            result.append(0)
    
    return result



class LinearRegression:

    def __init__(self):
        pass


    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        
    def predict(self, X:np.ndarray) -> list: 
        y_pre = X @ (np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y)
        y_pre = y_pre.reshape(-1, 1)
       
        return Labelize(y_pre)
