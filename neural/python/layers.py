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
        
        self.weights = .1 * np.random.randn(self.x.shape[-1], self.n_nodes)
        
        if not self.rand_weights:
            self.weights *= 0
        
        if self.rand_bias:
            self.bias = np.random.randn(self.n_nodes)
    
    def forward(self, derivative=False):
        z = np.dot(self.x, self.weights) + self.bias

        if derivative:
            return self.activation(z, derivative=True)
        
        if self.activation == None:
            return z
        
        self.output = self.activation(z)

        return self.output