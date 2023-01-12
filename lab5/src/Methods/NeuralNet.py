# 神经网络
# 调用了 SKL 库的内容

from sklearn.neural_network import MLPClassifier

class NeuralNetwork:
    
    def __init__(self, X, y):
        self.model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, ), random_state=1)
        self.X = X
        self.y = y
    
    
    
    def fit(self):
        self.model.fit(self.X, self.y)
        
        
    def predict(self, X):
        return self.model.predict(X)