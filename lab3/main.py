
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import networkx as nx





def random_Split_data(data: pd.DataFrame, rate = 0.75, random_seed: int = -1):
    m, n = data.shape
    if random_seed != -1:
        np.random.seed(abs(random_seed))
    _data = data.reindex(np.random.permutation(data.index)).copy()
   

    row_split = int(m * rate)
    X_train = _data.iloc[0: row_split, 0: n - 1].values
    y_train = _data.iloc[0: row_split, n - 1: ].values
    X_test = _data.iloc[row_split: m, 0: n - 1].values
    y_test = _data.iloc[row_split: m, n - 1: ].values
    
    return X_train, y_train, X_test, y_test


def show(times, loss, color = '#4169E1', start=0, end=1500):
    x_axis_data = list(range(times + 1))[start:end]
    y_axis_data = loss[start:end]
    plt.plot(x_axis_data, y_axis_data, color=color, alpha=0.8, linewidth=1)


      
        
# def create_graph(G, node:Node, pos={}, x=0, y=0, layer=1):

#     pos[node.id] = (x, y)
#     if node.l != -1:
#         G.add_edge(node.id, node.l.id)
#         l_x, l_y = x - 1 / 2 ** layer, y - 2
#         l_layer = layer + 1
#         create_graph(G, node.l, x=l_x, y=l_y, pos=pos, layer=l_layer)
#     if node.r != -1:
#         G.add_edge(node.id, node.r.id)
#         r_x, r_y = x + 1 / 2 ** layer, y - 2
#         r_layer = layer + 1
#         create_graph(G, node.r, x=r_x, y=r_y, pos=pos, layer=r_layer)
#     return (G, pos)


# def draw(node):   # 以某个节点为根画图
#     graph = nx.DiGraph()
#     graph, pos = create_graph(graph, node)
#     fig, ax = plt.subplots(figsize=(50, 100))  # 比例可以根据树的深度适当调节
#     nx.draw_networkx(graph, pos, ax=ax, node_size=1)
#     plt.show()


class Node:
    def __init__(self, index:list, id = -1, feature = -1, f_val = -1.) -> None:
        
        self.l = -1
        self.r = -1
        self.id = id
        self.w = -1
        
        self.index = index
        self.feature = feature
        self.f_val = f_val
        


class RegTree(object):
    def __init__(self, X:np.ndarray, y:np.ndarray, y_t:np.ndarray): # 初始化回归树
        self.X = X
        self.y = y
        self.m, self.n = X.shape
        
        self.g = 2 * (y_t - y)
        
        self.leaf_num = 0
        self.node_num = 0
        self.depth = 0
        
        self.R2_list = []
        
        

    def _get_best_split(self, node:Node): # 获得最佳feature和split
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
                print(node.id, feature, f_value, gain, index1, index2, self.X[index1, 0:1])
                print(self._get_score(node.index, self.leaf_num), self._get_score(index1, self.leaf_num + 1), self._get_score(index2, self.leaf_num + 1))
                if gain > max_gain:
                    max_gain = gain
                    max_feature = col
                    max_f_value = f_value
                    max_index1 = index1
                    max_index2 = index2
        print("the max is")
        print(max_feature, max_f_value, max_gain)
        print()        
        
        return max_feature, max_f_value, max_gain, max_index1, max_index2
                
            

    def _get_score(self, index:np.ndarray, T): # 获取某一划分的目标函数值  
        g = self.g[index, :].sum()
        score = (-0.5*g**2)/(2*len(index) + self.lamda) + self.gamma * T
        return score
    
    
    def _get_err(self):
        return (self.g.T @ self.g)[0][0] / 4
    
    
    
    def _fit(self, node:Node, depth = 1)->int:
        
        feature, f_value, gain, index1, index2 = self._get_best_split(node)
        print(feature, f_value)
        
        _feature = self.X[node.index, feature:feature+1].copy().reshape(1, -1)[0]
        f = list(set(_feature))
        
        if len(node.index) <= self.min_samples or len(f) <= self.min_feature_dif or depth >= self.max_depth or gain <= self.gain_delta:
            # This is a leaf
            new_w = -self.g[node.index, :].sum()/(2*len(node.index) + self.lamda)
            node.w = new_w
            
            # print(depth, new_w)
            self.leaf_num += 1
            
            if (self.if_silent == 0):
                
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
        
        self.R2_list.append(self.R2(self.predict(self.X), self.y))
    
        return max(self._fit(node.l, depth+1), self._fit(node.r, depth+1))
        
        
    def _set_parameters(self, default_parameters: dict):
        self.lamda = default_parameters['lamda']
        self.gamma = default_parameters['gamma']
        self.gain_delta = default_parameters['gain_delta']
        self.max_depth = default_parameters['max_depth']
        self.min_samples = default_parameters['min_samples']
        self.min_feature_dif = default_parameters['min_feature_dif']
        self.if_silent = default_parameters['if_silent']
        
    
    def fit(self, parameters: dict = {}):# 训练一棵回归树 

        default_parameters = {
            "lamda": 2,            # Hyperparameters
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
                
    
        
    def _predict(self, x:np.ndarray): # 预测一个样本
        
        node = self.root
    
        while node.l != -1 or node.r != -1:
            if x[0][node.feature] <= node.f_val:       
                node = node.l
            else:
                node = node.r
        return node.w
        

    def predict(self, X:np.ndarray): # 预测多条样本
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
    
    
    
# =======================

df = pd.read_csv('./lab3/test.data.txt',header=None)
X_train, y_train, X_test, y_test = random_Split_data(df, rate=1)
li = [0,1,2, 3]
print(np.array(li)[[3,2]])
print(X_train[[0, 1], :])
print(X_train[[0, 1], :])
print(X_train[[0, 1], :])
m, n = y_train.shape
y_t = np.zeros((m, n))


parameters = {
    "lamda": 0.1,            # Hyperparameters
    "gamma": 1e-6,          # Hyperparameters
    "gain_delta": 0,     # The minimum gain
    "max_depth": 3,
    "max_leaves": 100,
    "max_nodes": 1000,
    "min_samples": 0,      # Minimum number of samples on a leaf
    "min_feature_dif": 0,  # The minimum number of different value for current feature
}

tree = RegTree(X_train, y_train, y_t)
tree.fit(parameters)
print('===================')
print(tree.leaf_num, tree.depth)

pre = tree.predict(X_train)
# print(tree.root.l.l.w)
print(pre[0:10])
print(y_test[0:10])

print(tree.RMSE(pre, y_train))
print(tree.R2(pre, y_train))