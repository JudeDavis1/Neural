import numpy as np


def MSE(labels, outputs, derivative=False):
    # if type(labels) == np.ndarray or type(outputs) == np.ndarray:
    #     labels_shape, outputs_shape = np.product(labels.shape), np.product(outputs.shape)

    #     if labels_shape > outputs_shape:
    #         outputs = utils.pad_matrix(outputs, labels)
    #     else:
    #         labels = utils.pad_matrix(labels, outputs)

    if derivative:
        return np.mean(2 * (outputs - labels))
    
    return np.mean((outputs - labels) ** 2)

def categorical_crossentropy(labels, logits, derivative=False):
    if derivative:
        return np.mean(-(1 / logits).dot(labels))

    return np.mean(-np.log(logits).dot(labels))
