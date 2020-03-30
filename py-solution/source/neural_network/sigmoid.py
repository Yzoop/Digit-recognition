import numpy as np

def sigm(value):
    return 1 / (1 + np.exp(-value))