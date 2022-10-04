import numpy as np
class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg

    def sigmoid(self, x):
        """The logistic sigmoid function"""
        return 1.0 / (1 + np.exp(-x))

    def fit(self, X: np.ndarray, y: np.ndarray , lr=0.001, tol=1e-7, max_iter=1e7):
        """
        Fit the regression coefficients via gradient descent or other methods 
        """
        m, n = X.shape      # X is m rows n colums

        self.w = np.random.uniform(low=0.0, high=1.0, size=n).reshape(-1, 1)
        loss_list = []

        for i in range(int(max_iter)):
            temp = np.dot(X, self.w)
            loss = np.dot(-y.T, temp) + np.log(1 + np.exp(temp)).sum()
            loss_list.append(loss)
            self.w -= (lr * np.dot(X.T, self.sigmoid(temp) - y))
            if i > 2 and abs(loss_list[i] - loss_list[i - 1]) < tol:
                break
        
        return loss_list

        
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
        