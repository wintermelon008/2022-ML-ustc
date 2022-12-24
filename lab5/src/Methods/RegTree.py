# 回归树

import numpy as np

class Node:
    def __init__(self, index:list, id = -1, feature = -1, f_val = -1.) -> None:
        
        self.l = -1
        self.r = -1
        self.id = id
        self.w = -1
        
        self.index = index
        self.feature = feature
        self.f_val = f_val
        


class RegressionTree(object):
    def __init__(self, X:np.ndarray, y:np.ndarray, y_t:np.ndarray):
        # Initialize the regression tree
        self.X = X
        self.y = y
        self.m, self.n = X.shape
        
        self.g = 2 * (y_t - y)
        
        self.leaf_num = 0
        self.node_num = 0
        self.depth = 0
    
        self.err_list = []
        self.RMSE_list = []
    

    def _get_best_split(self, node:Node):
        # Get the best feature and split
        m = len(node.index)
        gain = -1
        feature = []
        feature_list = []
        
        max_gain = 0
        max_feature = -1
        max_f_value = -1
        
        max_index1 = []
        max_index2 = []
              
        for col in range(self.n):
            _feature = self.X[node.index, col:col+1].copy().reshape(1, -1)[0]

            feature = list(set(_feature))
            feature.sort()
            
            for f_value in feature:
                _index1 = np.where((self.X[node.index, col:col+1] <= f_value).all(axis=1))[0]
                index1 = np.array(node.index)[_index1]
                _index2 = np.where((self.X[node.index, col:col+1] > f_value).all(axis=1))[0]
                index2 = np.array(node.index)[_index2]
                gain = self._get_score(node.index, self.leaf_num) - self._get_score(index1, self.leaf_num + 1) - self._get_score(index2, self.leaf_num + 1)

                if gain > max_gain:
                    max_gain = gain
                    max_feature = col
                    max_f_value = f_value
                    max_index1 = index1
                    max_index2 = index2
                
        return max_feature, max_f_value, max_gain, max_index1, max_index2
                
            

    def _get_score(self, index:np.ndarray, T):
        # Get the objective function value of a partition  
        g = self.g[index, :].sum()
        score = (-0.5*g**2)/(2*len(index) + self.lamda) + self.gamma * T
        return score
    
    
    def _get_err(self, pre: np.ndarray):
        y = pre - self.y
        return (y.T @ y)[0][0]
    
    
    def _fit(self, node:Node, depth = 1)->int:
        cur_pre = self.predict(self.X)
        self.RMSE_list.append(self.RMSE(cur_pre, self.y))
        self.err_list.append(self._get_err(cur_pre))
        
        feature, f_value, gain, index1, index2 = self._get_best_split(node)
        
        _feature = self.X[node.index, feature:feature+1].copy().reshape(1, -1)[0]
        f = list(set(_feature))
        
        if len(node.index) <= self.min_samples or len(f) <= self.min_feature_dif or depth >= self.max_depth or gain <= self.gain_delta:
            # This is a leaf
            
            new_w = -self.g[node.index, :].sum()/(2*len(node.index) + self.lamda)
            node.w = new_w
            self.leaf_num += 1
            
            if (self.if_silent == 0):
                print(depth, new_w)
                if len(node.index) <= self.min_samples:
                    print("stop reason: no enough samples")
                elif len(f) <= self.min_feature_dif:
                    print("stop reason: no enough feature values")
                elif depth >= self.max_depth:
                    print("stop reason: depth reaches limit")
                elif gain <= self.gain_delta:
                    print("stop reason: gain less than [gain_delta]")
                
            return depth
        
        node.feature = feature
        node.f_val = f_value
        node.l = Node(index1, id=self.node_num)
        node.r = Node(index2, id=self.node_num + 1)
        self.node_num += 2

        return max(self._fit(node.l, depth+1), self._fit(node.r, depth+1))
        
        
    def _set_parameters(self, default_parameters: dict):
        self.lamda = default_parameters['lamda']
        self.gamma = default_parameters['gamma']
        self.gain_delta = default_parameters['gain_delta']
        self.max_depth = default_parameters['max_depth']
        self.min_samples = default_parameters['min_samples']
        self.min_feature_dif = default_parameters['min_feature_dif']
        self.if_silent = default_parameters['if_silent']
        
    
    def fit(self, parameters: dict = {}):
        # Train a regression tree 
        default_parameters = {
            "lamda": 2,             # Hyperparameters
            "gamma": 1e-6,          # Hyperparameters
            "gain_delta": 0,        # The minimum gain
            "max_depth": 7,
            "max_leaves": 100,
            "max_nodes": 1000,
            "min_samples": 10,      # Minimum number of samples on a leaf
            "min_feature_dif": 5,   # The minimum number of different value for current feature
            "if_silent": 1
        }
        
        for key in parameters.keys():
            if default_parameters.__contains__(key):
                default_parameters[key] = parameters[key]
        self._set_parameters(default_parameters)
        
        index = np.arange(self.m)
        node = Node(index, 0)
        self.root = node
        self.node_num = 1
        self.depth = self._fit(node)
                
    
    def _predict(self, x:np.ndarray): 
        # Predict a sample
        node = self.root
        while node.l != -1 or node.r != -1:
            if x[0][node.feature] <= node.f_val:       
                node = node.l
            else:
                node = node.r
        return node.w
        

    def predict(self, X:np.ndarray): 
        # Predict multiple samples
        m, _ = X.shape
        pre = []
        for i in range(m):
            prei = self._predict(X[i].reshape(1, -1))
            pre.append(prei)
        
        return np.array(pre).reshape(m, 1)
    
    
    def RMSE(self, pre:np.ndarray, val:np.ndarray):
        y = val - pre
        m, _ = y.shape
        return np.sqrt((y.T @ y) / m)[0][0]
    
    def R2(self, pre:np.ndarray, val:np.ndarray):
        var = np.var(val)
        return(1 - (self.RMSE(pre, val) ** 2)/ var)