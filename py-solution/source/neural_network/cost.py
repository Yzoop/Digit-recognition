import numpy as np

def neural_cost_function(neural_network_gradient: list, input_layer_size,
                         hidden_layer_size, num_labels, X, y, lambda_koef):
    """
    Function: neural_cost_function()

    Description: Implements the neural network cost function for a two layer
                 neural network which performs classification

    Parameters:  J, grad = neural_cost_function(neural_network_gradient, hidden_layer_size, num_labels, ...
                 X, y, lambda_koef) computes the cost and gradient of the neural network. The
                 parameters for the neural network are "unrolled" into the vector
                 nn_params and need to be converted back into the weight matrices.

    Returns: The returned parameter grad should be a "unrolled" vector of the
             partial derivatives of the neural network.
    """
    J = 0
    Theta1_grad = np.zeros(neural_network_gradient[0].shape)
    Theta2_grad = np.zeros(neural_network_gradient[1].shape)

    # Part 1: Feedforward the neural network and return the cost in the
    # variable J. After implementing Part 1, you can verify that your
    # cost function computation is correct by verifying the cost

    # forward proparagition
