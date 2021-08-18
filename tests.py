import random
import numpy as np

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
        neural.layers.Dense(len(y), activation=neural.activations.sigmoid)
    ])
    
    model.compile(neural.optimizers.SGD(lr=LR), neural.losses.MSE)
    model.fit(x, y, epochs=EPOCHS)

    output = model.predict(x)
    print(output)


if __name__ == '__main__':
    main()