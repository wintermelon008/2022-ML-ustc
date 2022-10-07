import numpy as np
class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg

    def sigmoid(self, x):
        """The logistic sigmoid function"""
        return 1.0 / (1 + np.exp(-x))

    def fit(self, X: np.ndarray, y: np.ndarray , lr=0.001, tol=1e-7, max_iter=1e7, method='gradient'):
        """
        Fit the regression coefficients via gradient descent or other methods 
        """
        assert method in ['gradient', 'newton'], "method must be 'gradient' or 'newton', but got {}".format(method)

        if method == 'gradient':
            return self.gradient(X, y, lr, tol, max_iter)
        elif method == 'newton':
            return self.newton(X, y, lr, tol, max_iter)



    def gradient(self, X: np.ndarray, y: np.ndarray , lr=0.001, tol=1e-7, max_iter=1e7):
        
        m, n = X.shape    
        self.w = np.random.uniform(low=0.0, high=1.0, size=n).reshape(-1, 1)
        loss_list = []

        for i in range(int(max_iter)):
            temp = np.dot(X, self.w)
            loss = np.dot(-y.T, temp) + np.log(1 + np.exp(temp)).sum()
            loss_list.append(loss[0][0])
            delta1 = (np.dot(X.T, self.sigmoid(temp) - y))
            if i > 2 and np.dot(delta1.T, delta1) < tol:
                break
            self.w -= lr * delta1
        
        return i, loss_list

    

    def newton(self, X: np.ndarray, y: np.ndarray , lr=0.001, tol=1e-7, max_iter=1e7):
        m, n = X.shape    

        self.w = np.random.uniform(low=0.0, high=1.0, size=n).reshape(-1, 1)
        loss_list = []

        for i in range(int(max_iter)):
            temp = np.dot(X, self.w)
            loss = np.dot(-y.T, temp) + np.log(1 + np.exp(temp)).sum()
            loss_list.append(loss[0][0])
            diag_p1 = np.diag(self.sigmoid(temp).reshape(-1)) 
            diag = np.dot(diag_p1, np.identity(m) - diag_p1)
            delta1 = (np.dot(X.T, self.sigmoid(temp) - y))
            
            if i > 2 and np.dot(delta1.T, delta1) < tol:
                break
            delta2 = np.dot(np.dot(X.T, diag), X)
            self.w -= np.dot(np.linalg.inv(delta2), delta1)
        
        return i, loss_list
        

        
    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """

        m, n = X.shape
        result = []
        temp = np.dot(X, self.w)
        for i in range(m):
            if temp[i] >= 0.5:
                result.append(1)
            else:
                result.append(0)
        
        return result
        