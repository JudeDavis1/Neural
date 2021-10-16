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
        
<<<<<<< HEAD
        dw = np.mean(np.dot(layer.x.T, self._d_activation) * self._d_loss, axis=1)
        db = np.sum(self._d_activation * self._d_loss, axis=0)

        layer.weights = (layer.weights.T - (self.lr * dw.T)).T
=======
        dw = np.mean(np.dot(layer.x.T, self._d_activation) * self._d_loss)
        db = np.mean(self._d_activation * self._d_loss)

        layer.weights = layer.weights - (self.lr * dw.T)
>>>>>>> cfa4ae9c10705b1246487b800743c273394def00
        layer.bias = layer.bias - (self.lr * db.T)
