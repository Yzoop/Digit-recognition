import numpy as np

SMALLER_KOEF = 0.01

def __get_initialized_gradient__(width, height):
    return np.random.randn(width, height) * SMALLER_KOEF #so as to fast learning algotihm

def initial_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    W1 = __get_initialized_gradient__(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = __get_initialized_gradient__(n_y, n_h)
    b2 = np.zeros((1, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters