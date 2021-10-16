import random
import numpy as np
<<<<<<< HEAD
import matplotlib.pyplot as plt
import sys
import neural


LR = .01
N_HIDDEN = 1
EPOCHS = int(sys.argv[1])

def main():
    x = np.random.randn(3, 4)
    y = np.random.random(len(x))

    model = neural.models.NN([
        neural.layers.Dense(7, activation=None, rand_weights=False),
=======

import neural

LR = 0.01
N_HIDDEN = 1
EPOCHS = 1000

def main():
    x = 1 + np.random.random((3, 4))
    y = np.random.random(len(x))

    model = neural.models.NN([
        neural.layers.Dense(7, activation=None),
        neural.layers.Dense(2, activation=None),
>>>>>>> cfa4ae9c10705b1246487b800743c273394def00
        neural.layers.Dense(len(y), activation=neural.activations.sigmoid)
    ])
    
    model.compile(neural.optimizers.SGD(lr=LR), neural.losses.MSE)
    model.fit(x, y, epochs=EPOCHS)

    output = model.predict(x)
    print(output)


if __name__ == '__main__':
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> cfa4ae9c10705b1246487b800743c273394def00
