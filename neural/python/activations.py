import numpy as np

def linear(z, derivative=False):
    if derivative:
        return 1
    return z

def sigmoid(z, derivative=False):
    f = 1 / (1 + np.exp(-z))

    if derivative:
        return f * (1 - f)
    
    return f

def relu(z, derivative=False, epsilon=1e-16):
    f = np.maximum(z + epsilon, 0)

    if derivative:
        if np.sum(f) != 0:
            return 0
        else:
            return 1
    
    return f

def softmax(z, derivative=False):
    f = np.exp(z - np.max(z, axis=1, keepdims=True))
    f /= np.sum(f, axis=1, keepdims=True)

    if derivative:
        return f * (1 - f)
    
    return f

def tanh(z, derivative=False):
    if derivative:
        return (1 / np.cosh(z)) ** 2
    
    return np.tanh(z)