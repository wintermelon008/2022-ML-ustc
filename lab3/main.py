
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



class Node:
    def __init__(self, index:list, id = -1) -> None:
        self.l = -1
        self.r = -1
        self.id = id
        self.index = index
        
      
        
def create_graph(G, node:Node, pos={}, x=0, y=0, layer=1):

    pos[node.id] = (x, y)
    if node.l != -1:
        G.add_edge(node.id, node.l.id)
        l_x, l_y = x - 1 / 2 ** layer, y - 2
        l_layer = layer + 1
        create_graph(G, node.l, x=l_x, y=l_y, pos=pos, layer=l_layer)
    if node.r != -1:
        G.add_edge(node.id, node.r.id)
        r_x, r_y = x + 1 / 2 ** layer, y - 2
        r_layer = layer + 1
        create_graph(G, node.r, x=r_x, y=r_y, pos=pos, layer=r_layer)
    return (G, pos)


def draw(node):   # 以某个节点为根画图
    graph = nx.DiGraph()
    graph, pos = create_graph(graph, node)
    fig, ax = plt.subplots(figsize=(50, 100))  # 比例可以根据树的深度适当调节
    nx.draw_networkx(graph, pos, ax=ax, node_size=1)
    plt.show()


class RegTree(object):
    def __init__(self, X:np.ndarray, y:np.ndarray, y_t:np.ndarray): # 初始化回归树
        self.X = X
        self.y = y
        self.g = 2 * (y_t - y)
        self.m, self.n = X.shape
        self.w = []
        self.T = 0
        self.nodes = 0
        self.depth = 0
        
        

    def _get_best_split(self, node:Node): # 获得最佳feature和split
        m = len(node.index)
        feature = []
        feature_list = []
        max_gain = -np.inf
        max_feature = -1
        max_f_value = -1
        
        
        
        for col in range(self.n):
            _feature = self.X[node.index, col:col+1].copy().reshape(1, -1)[0]

            feature = list(set(_feature))
            feature.sort()
            
            for f_value in feature:
                index1 = np.where((self.X[node.index, col:col+1] <= f_value).all(axis=1))[0]
                index2 = np.where((self.X[node.index, col:col+1] > f_value).all(axis=1))[0]
                gain = self._get_score(node.index) - self._get_score(index1) - \
                       self._get_score(index2)
                # print(rowindex[0])
                if gain > max_gain:
                    max_gain = gain
                    max_feature = col
                    max_f_value = f_value
                
        return max_feature, max_f_value, gain
                
            

    def _get_score(self, index:np.ndarray): # 获取某一划分的目标函数值  
        g = self.g[index, :].sum()
        score = (-0.5*g**2)/(2*len(index) + self.lamda) + self.gamma * 1
        return score
    
    
    def _fit(self, node:Node, depth = 0):
        feature, f_value, gain = self._get_best_split(node)
        
        _feature = self.X[node.index, feature:feature+1].copy().reshape(1, -1)[0]

        f = list(set(_feature))
        
        # if gain < self.gain_delta:
        if len(node.index) < 50 or len(f) < 3 or depth > 20:
            # Update w
            best_w = -self.g[node.index, :].sum()/(2*len(node.index) + self.lamda)
            self.w.append(best_w)
            self.T += 1
            return depth
        
        index1 = np.where((self.X[node.index, feature:feature+1] <= f_value).all(axis=1))[0]
        index2 = np.where((self.X[node.index, feature:feature+1] > f_value).all(axis=1))[0]
        
        node.l = Node(index1, self.nodes)
        self.nodes += 1
        node.r = Node(index2, self.nodes)
        self.nodes += 1
        
        return max(self._fit(node.l, depth+1), self._fit(node.r, depth+1))
        
        

    
    
    def fit(self, lamda = 0.1, gamma = 0.01, gain_delta = 1e-2):# 训练一棵回归树 
        self.lamda = lamda
        self.gamma = gamma
        self.gain_delta = gain_delta
        
        index = np.arange(self.m)
        node = Node(index, self.nodes)
        self.nodes += 1
        self.depth = self._fit(node)
        print(self.w)
        # print("finish")
        # draw(node)
        
            
            
        
    def _predict(self,): # 预测一个样本 
        pass
    def predict(self,): # 预测多条样本
        pass
    
    
    
# =======================

df = pd.read_csv('./lab3/train.data.txt',header=None)
X_train, y_train, X_test, y_test = random_Split_data(df, rate=0.7)
m, n = y_train.shape
y_t = np.zeros((m, n))

tree = RegTree(X_train, y_train, y_t)
tree.fit()
