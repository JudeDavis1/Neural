import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


class NN:

    def __init__(self, layers=[]):
        self.layers = layers
        self.all_losses = []
        self.started_training = False
    
    def predict(self, x):
        if not self.started_training:
            self._initialize(x)
        else:
            for i in tqdm(range(1, len(self.layers))):
                self.layers[i].x = self.layers[i - 1].forward()
        
        return self.layers[-1].forward()
    
    def compile(self, optimizer, loss):
        self.loss = loss
        self.optimizer = optimizer
    
    def fit(self, x_train, y_train, epochs=100):
        self.epochs = epochs
        self.outputs = self.predict(x_train)
        self.optimizer.y = y_train
        self.started_training = True

        pbar = tqdm(range(epochs))

        for epoch in pbar:
            self.outputs = self.predict(x_train)
            loss = self.loss(y_train, self.outputs)

            pbar.set_description(f'Loss: {loss}')
            self.all_losses.append(loss)

            for i in range(len(self.layers)):
                layer_output = self.layers[i].forward()

                self.optimizer.output = layer_output
                self.optimizer._d_loss = self.loss(y_train, layer_output.T, derivative=True)
                
                if self.layers[i].activation == None:
                    self.optimizer._d_activation = 1
                else:
                    self.optimizer._d_activation = self.layers[i].forward(derivative=True)
                
                if i == 0:
                    self.optimizer.x = x_train
                else:
                    self.optimizer.x = self.layers[i - 1].forward()
                self.optimizer._propagate(self.layers[i])
    
    def plot_loss(self):
        plt.title('Evolution of Loss over epochs (lower is better)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(range(self.epochs), self.all_losses)

        plt.show()
    
    def _get_batches(self, x, batch_size):
        batches = [batch for batch in np.pad(x, (0, len(x) % batch_size))]
        return batches
    
    def _initialize(self, x):
        self.layers[0].x = x
        self.layers[0].build(x)
        self.layers[0].id = 0

        for i in range(1, len(self.layers)):
            self.layers[i].id = i
            self.layers[i].build(self.layers[i - 1].forward())
