import numpy as np

from .layers import Dense

class SGD:

    def __init__(self, lr=.01):
        self.lr = lr
        
        self.x = None
        self.y = None
        self.output = None
        self._d_loss = None
        self._d_activation = None
    
    def _propagate(self, layer: Dense):
        # gradient descent with partial derivatives
        if type(self._d_activation) == int:
            self._d_activation = np.array(self._d_activation)
        
        dw = np.mean(np.dot(layer.x.T, self._d_activation) * self._d_loss)
        db = np.mean(self._d_activation * self._d_loss)

        layer.weights = layer.weights - (self.lr * dw.T)
        layer.bias = layer.bias - (self.lr * db.T)
