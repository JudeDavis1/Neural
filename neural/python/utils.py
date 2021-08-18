import pickle
import numpy as np

def pad_matrix(a, b):
    r = np.zeros(b.shape)
    if len(a.shape) == 1 or len(b.shape) == 1:
        return np.pad(a, (0, abs(len(b) - len(a))))

    r[:a.shape[0], :a.shape[1]] = a

    return r

def pad_matrix_equal(a: np.ndarray, b: np.ndarray):
    if np.product(a.shape) < np.product(b.shape):
        return (pad_matrix(a, b), b)
    else:
        return (a, pad_matrix(b, a))

def save(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))

def load(filename):
    return pickle.load(open(filename, 'rb'))
