# XGBoost

from xgboost import XGBClassifier as XGBC
import time
import pickle

class XGBoost(object):
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        
        
    def _set_parameters(self, default_parameters: dict):
        self.if_silent = default_parameters['if_silent']
        
        self.model = XGBC(
            learning_rate = default_parameters['learning_rate'],
            min_child_weight = default_parameters['min_child_weight'],
            max_depth = default_parameters['max_depth'],
            gamma = default_parameters['gamma'],
            subsample = default_parameters['subsample'],
            max_delta_step = default_parameters['max_delta_step'],
            colsample_bytree = default_parameters['colsample_bytree'],
            reg_lambda = default_parameters['reg_lambda'],
            reg_alpha = default_parameters['reg_alpha'],
            objective = default_parameters['objective'],
            num_class = default_parameters['num_class'],
            n_estimators = default_parameters['n_estimators'],
            seed = default_parameters['seed'],          
        )
    
    def fit(self, parameters:dict = {}):
        
        default_parameters = {
            "learning_rate": 0.3,       # 如同学习率
            "min_child_weight": 1, 
            # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
            #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
            # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            "max_depth": 8,             # 构建树的深度，越大越容易过拟合
            "gamma": 0,                 # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
            "subsample": 1,             # 随机采样训练样本 训练实例的子采样比
            "max_delta_step": 0,        # 最大增量步长，我们允许每个树的权重估计。
            "colsample_bytree": 1,      # 生成树时进行的列采样 
            "reg_lambda": 1,            # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            "reg_alpha": 0,             # L1 正则项参数       
            "objective": 'multi:softmax',   # 多分类的问题 指定学习任务和相应的学习目标 binary：logitraw
            "num_class": 4,             # 类别数
            "n_estimators": 50,         # 树的个数
            "seed": 1000,                # 随机种子     
            "if_silent": 1,
        }
        for key in parameters.keys():
            if default_parameters.__contains__(key):
                default_parameters[key] = parameters[key]
        self._set_parameters(default_parameters)
        
        self.model.fit(self.X, self.y)
        
    
    def predict(self, X):
        return self.model.predict(X)
    
    
    def save(self, filename = None):
        if filename == None:
            t_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
            filename = "./model/Model_XGBoost" + "_" + t_str + ".dat"
            
        pickle.dump(self, open(filename, "wb"))
