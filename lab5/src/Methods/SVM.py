# 支持向量机
import numpy as np


def my_max(a, b):
    return a if a > b else b

def find_zero(a, b):
    return 0 if a == 0 else b

def array_max(a:np.ndarray, b:np.ndarray)->np.ndarray:
    func_ = np.frompyfunc(my_max, 2, 1)
    return(func_(a, b))

def array_find0(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    func_ = np.frompyfunc(find_zero, 2, 1)
    return(func_(a, b))

# The above function implements the corresponding operation of each dimension of the vector.


class SupportVectorMachine:

    def __init__(self, X:np.ndarray, y:np.ndarray):
        """
        ---
        Args:
        ---
            X (np.ndarray): 
                Data characteristics
            y (np.ndarray): 
                Data category
        """
        self.X = X
        self.y = y
        self.m, self.n = self.X.shape 
        self.w = np.zeros((self.n + 1, 1))
        
        
        unique, count=np.unique(self.y, return_counts=True)
        data_count = dict(zip(unique, count))
        
        keys = []
        for key in data_count:
            keys.append(key)
                
        self.neg = keys[0]
        self.pos = keys[1]
        self.y[self.y == self.neg] = -1
        self.y[self.y == self.pos] = 1


    def fit(self, gamma=0.25, lr=0.002, tol=1e-4, max_times=500, ifsilent=True):
        """
        ---
        Args:
        ---
            gamma: 
                Regularization parameters. 
                By default 0.25
            lr: 
                Learning rate. 
                By default 0.002
            max_times: 
                The maximum times of training iterations. 
                By default 500
            tol: 
                The minimum amount of change for the adjacent two iterations of the loss function.
                By default 1e-4
            ifsilent: 
                Whether toprint training process information. 
                If False, the program will print the value of the loss function during iteration. 
                By default True
        """

        # Preprocess the training data
        temp_1 = np.ones((self.m, 1))
        X_hat:np.ndarray = np.c_[self.X, temp_1]

        temp_0 = np.zeros((self.m, 1))
        loss_list = []
        y_diag = np.diag(self.y.reshape(-1))
        
        info_gap = max_times / 20
        
        # Start iterating
        if ifsilent:
            for times in range(max_times):
                xi = array_max(temp_0, 1 - (y_diag @ X_hat @ self.w))
                loss = 0.5 * (self.w.T @ self.w)[0][0] + gamma * (xi.sum())
            
                y_bar = array_find0(xi , self.y)
                delta_1 = self.w - gamma * (X_hat.T @ y_bar)
                
                if times >= 2 and abs(loss_list[-1] - loss) < tol:
                    loss_list.append(loss)
                    break

                self.w = self.w - lr * delta_1
                loss_list.append(loss)
        else:
            for times in range(max_times):
                xi = array_max(temp_0, 1 - (y_diag @ X_hat @ self.w))
                loss = 0.5 * (self.w.T @ self.w)[0][0] + gamma * (xi.sum())
                
                y_bar = array_find0(xi , self.y)
                delta_1 = self.w - gamma * (X_hat.T @ y_bar)
                
                if times >= 2 and abs(loss_list[-1] - loss) < tol:
                    loss_list.append(loss)
                    break

                if (times % info_gap == 0):
                    print("Current times: {}, loss is {}".format(times, loss))

                self.w = self.w - lr * delta_1
                loss_list.append(loss)

        return loss_list, times
            

        
    def predict(self, X:np.ndarray) -> np.ndarray:
        m, n = X.shape
        temp_1 = np.ones(m)
        X_hat:np.ndarray = np.c_[X, temp_1]
        temp = X_hat @ self.w
        ans = []
        for i in range(m):
            if temp[i] > 0:
                ans.append(self.pos)
            else:
                ans.append(self.neg)
        return np.array(ans).reshape(-1, 1)