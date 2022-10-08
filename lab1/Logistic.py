import numpy as np

class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True, random_seed: int = -1, reg_param = 0.01):
        regularization_list = ['l0', 'l1', 'l2']
        err_msg = "penalty must be 'l0', 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in regularization_list, err_msg

        if random_seed == -1:
            self.ifset_random_seed = False
        else:
            self.ifset_random_seed = True

        self.random_seed = abs(random_seed) 

        self.reg_1param = 0
        self.reg_2param = 0

        if penalty == 'l1':
            self.reg_1param = reg_param
        elif penalty == 'l2':
            self.reg_2param = reg_param



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
        if self.ifset_random_seed:
            np.random.seed(self.random_seed)
        self.w = np.random.uniform(low=0.0, high=1.0, size=n).reshape(-1, 1)
        loss_list = []

        for i in range(int(max_iter)):
            temp = np.dot(X, self.w)
            loss = np.dot(-y.T, temp) + np.log(1 + np.exp(temp)).sum()

            loss_list.append(loss[0][0])

            delta1 = (np.dot(X.T, self.sigmoid(temp) - y)) + self.reg_1param * np.sign(self.w) + self.reg_2param * 2 * self.w
            
            if i > 2 and np.dot(delta1.T, delta1) < tol:
                break
            self.w -= lr * delta1
        
        return i, loss_list

    

    def newton(self, X: np.ndarray, y: np.ndarray , lr=0.001, tol=1e-7, max_iter=1e7):
        m, n = X.shape
        if self.ifset_random_seed:
            np.random.seed(self.random_seed)
        self.w = np.random.uniform(low=0.0, high=1.0, size=n).reshape(-1, 1)
        loss_list = []

        for i in range(int(max_iter)):
            temp = np.dot(X, self.w)

            loss = np.dot(-y.T, temp) + np.log(1 + np.exp(temp)).sum()

            loss_list.append(loss[0][0])

            diag_p1 = np.diag(self.sigmoid(temp).reshape(-1)) 
            diag = np.dot(diag_p1, np.identity(m) - diag_p1)
            delta1 = (np.dot(X.T, self.sigmoid(temp) - y)) + self.reg_1param * np.sign(self.w) + self.reg_2param * 2 * self.w
            
            if i > 2 and np.dot(delta1.T, delta1) < tol:
                break
            delta2 = np.dot(np.dot(X.T, diag), X) + self.reg_2param * 2 * np.identity(n)
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
        