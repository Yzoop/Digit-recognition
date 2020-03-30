import numpy as np

SMALLER_KOEF = 0.01

def get_initialized_gradient(width, height):
    return np.random.randn(width, height) * SMALLER_KOEF #so as to fast learning algotihm