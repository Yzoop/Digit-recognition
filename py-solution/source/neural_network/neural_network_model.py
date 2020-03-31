# GRADED FUNCTION: nn_model
import numpy as np
import source.neural_network.__gradient__ as __gradient__
import source.neural_network.__gradient_descent__ as __gradient_descent__
import source.neural_network.__propagations__ as __propagations__
import source.neural_network.__cost__ as __cost__


last_cost = 0

"""
Here we use neural network with 2 hidden layers

Images we use: 20x20 (it's possible to change later)
"""
def nn_model(X, Y, n_h, num_iterations=800, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    global last_cost
    NUMBER_OF_LABELS = 10  # from 0 to 9
    INPUT_LAYER_SIZE = 400  # as 20 * 20 = 400

    # Initialize parameters
    parameters =  __gradient__.initial_parameters(INPUT_LAYER_SIZE, n_h, NUMBER_OF_LABELS)
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = __propagations__.forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        last_cost = cost = __cost__.compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = __propagations__.backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = __gradient_descent__.update_parameters(parameters, grads)

        # Print the cost every 1000 iterations
        if print_cost and i % 50 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters

def get_last_cost():
    return last_cost