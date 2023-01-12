# 决策树模型
# 调用 skl 的模型

from sklearn.tree import DecisionTreeClassifier
import pickle
import time

class DecisionTree:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
        
    def _set_parameters(self, default_parameters: dict):
        self.if_silent = default_parameters['if_silent']
        self.tree = DecisionTreeClassifier(
            criterion=default_parameters['criterion'],
            splitter = default_parameters['splitter'],
            min_samples_split = default_parameters['min_samples_split'],
            max_depth = default_parameters['max_depth'],
            min_samples_leaf = default_parameters['min_samples_leaf'],
            max_leaf_nodes = default_parameters['max_leaf_nodes'],
            min_impurity_decrease = default_parameters['min_impurity_decrease']
        )
        
    
    def fit(self, parameters: dict = {}):
        default_parameters = {
            "criterion": "entropy",    # 选择特征的标准，分为 "gini" 和 "entropy"
            "splitter": "best",        # 特征划分标准，分为 "best" 和 "random"
            "max_depth": 7,
            "min_samples_split": 1,    # 叶子最小样本数
            "min_samples_leaf": 2,     # 划分最小样本数
            "max_leaf_nodes": 1000,    # 最大叶节点个数
            "min_impurity_decrease": 0.0, # 最小划分不纯度减少量
            "if_silent": 1
        }
        
        for key in parameters.keys():
            if default_parameters.__contains__(key):
                default_parameters[key] = parameters[key]
        self._set_parameters(default_parameters)
        
        self.tree.fit(self.X, self.y)
        
        
    def predict(self, X):
        return self.tree.predict(X)  
    
    
    
    def save(self, filename = None):
        if filename == None:
            t_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
            filename = "./model/Model_DecTree" + "_" + t_str + ".dat"
            
        pickle.dump(self, open(filename, "wb"))