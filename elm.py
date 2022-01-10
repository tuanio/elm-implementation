import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from utils import relu, sigmoid

class ELMBase:
    def __init__(self, n_hiddens=128, random_state=12):
        self.n_hiddens = n_hiddens
        self.rs = np.random.RandomState(random_state)
    
    
class ELMRegressor(ELMBase):
    def __init__(self, n_hiddens=128, random_state=12):
        ELMBase.__init__(self, n_hiddens, random_state)
        self.m = 1
        self.activation = lambda x: x # linear activation
        # self.activation = relu
        
    def fit(self, X, y):
        self.W = self.rs.normal(size=(X.shape[1], self.n_hiddens))
        self.b = self.rs.normal(size=(self.n_hiddens))
        y = y.reshape(-1, 1)
        
        H = self.activation(X.dot(self.W) + self.b)
        self.Beta = np.linalg.pinv(H).dot(y)
        return self
    
    def predict(self, X):
        H = self.activation(X.dot(self.W) + self.b)
        dot_product = H.dot(self.Beta)
        return dot_product
    
    
class ELMClassifier(ELMBase):
    def __init__(self, n_hiddens=128, random_state=12):
        ELMBase.__init__(self, n_hiddens, random_state)
        self.activation = relu
        self.output_activation = softmax
        self.encoder = OneHotEncoder()
        
    def fit(self, X, y):
        self.W = self.rs.normal(size=(X.shape[1], self.n_hiddens))
        self.b = self.rs.normal(size=(self.n_hiddens))
        y = self.encoder.fit_transform(y.reshape(-1, 1)).toarray()
        
        H = self.activation(X.dot(self.W) + self.b)
        self.Beta = np.linalg.pinv(H).dot(y)
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def predict_proba(self, X):
        H = self.activation(X.dot(self.W) + self.b)
        dot_product = H.dot(self.Beta)
        return self.output_activation(dot_product)