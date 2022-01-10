import numpy as np

def relu(x):
    return x * (x > 0)

def sigmoid(x):
    return x / (1 + np.exp(x))
    