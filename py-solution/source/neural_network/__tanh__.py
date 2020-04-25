import numpy as np

def tanh(Z):
    return np.tanh(Z)

def tanh_backward(W2, dZ2, A1):
    return np.multiply(np.dot(W2.transpose(), dZ2), (1 - np.power(A1, 2)))