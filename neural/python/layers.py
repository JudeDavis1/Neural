import numpy as np

from . import activations

class Dense:

    def __init__(self,
                 n_nodes: int,
                 x: np.ndarray=None,
                 activation=activations.relu,
                 rand_weights=True,
                 rand_bias=True):

        self.x = x
        self.id = 0
        self.n_nodes = n_nodes
        self.activation = activation

        self.rand_bias = rand_bias
        self.rand_weights = rand_weights
    
    def build(self, x):
        self.x = x
        
<<<<<<< HEAD
        self.weights = .35 * np.random.randn(self.x.shape[-1], self.n_nodes)
=======
        self.weights = .1 * np.random.randn(self.x.shape[-1], self.n_nodes)
>>>>>>> cfa4ae9c10705b1246487b800743c273394def00
        
        if not self.rand_weights:
            self.weights *= 0
        
        if self.rand_bias:
<<<<<<< HEAD
            self.bias = .1 * np.random.randn(self.n_nodes)
=======
            self.bias = np.random.randn(self.n_nodes)
>>>>>>> cfa4ae9c10705b1246487b800743c273394def00
    
    def forward(self, derivative=False):
        z = np.dot(self.x, self.weights) + self.bias

        if derivative:
            return self.activation(z, derivative=True)
        
        if self.activation == None:
            return z
        
        self.output = self.activation(z)

<<<<<<< HEAD
        return self.output
=======
        return self.output
>>>>>>> cfa4ae9c10705b1246487b800743c273394def00
