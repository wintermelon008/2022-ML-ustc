# 神经网络
# 调用了 SKL 库的内容

from sklearn.neural_network import MLPClassifier

class NeuralNetwork:
    
    def __init__(self, X, y):
        self.model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, ), random_state=1)
        self.X = X
        self.y = y
    
    
    def _set_parameters(self, default_parameters: dict):
        
        self.model = MLPClassifier(
            hidden_layer_sizes = default_parameters["hidden_layer_sizes"],
            activation = default_parameters["activation"],
            alpha = default_parameters["alpha"],
            max_iter = default_parameters["max_iter"],
            tol = default_parameters["tol"],
            verbose = default_parameters["if_silent"] == False,
            early_stopping = default_parameters["early_stopping"],
            validation_fraction = default_parameters["validation_fraction"],
        )
    
    
    def fit(self, parameters: dict = {}):
        default_parameters = {
            "hidden_layer_sizes": (100, ),
            "activation": 'relu', # 激活函数, ‘identity’，‘logistic’，‘tanh’，‘relu’
            "alpha": 0.0001, # 可选，默认为0.0001。L2惩罚（正则化项）参数。
            "max_iter": 200, # 默认值200。最大迭代次数。solver迭代直到收敛（由’tol’确定）或这个迭代次数。对于随机解算器（‘sgd’，‘adam’），请注意，这决定了时期的数量（每个数据点的使用次数），而不是梯度步数。
            "tol": 1e-4, #默认1e-4 优化的容忍度，容差优化。当n_iter_no_change连续迭代的损失或分数没有提高至少tol时，除非将learning_rate设置为’adaptive’，否则认为会达到收敛并且训练停止。
            "if_silent": True, #是否将进度消息打印到stdout。
            "early_stopping": False, # 当验证评分没有改善时，是否使用提前停止来终止培训。如果设置为true，它将自动留出10％的训练数据作为验证，并在验证得分没有改善至少为n_iter_no_change连续时期的tol时终止训练。仅在solver ='sgd’或’adam’时有效
            "validation_fraction": 0.1, # 将训练数据的比例留作早期停止的验证集。必须介于0和1之间。仅在early_stopping为True时使用
        }
        for key in parameters.keys():
            if default_parameters.__contains__(key):
                default_parameters[key] = parameters[key]
        self._set_parameters(default_parameters)
        
        
        self.model.fit(self.X, self.y)
        
        
    def predict(self, X):
        return self.model.predict(X)