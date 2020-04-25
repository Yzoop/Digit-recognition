# GRADED FUNCTION: nn_model
import numpy as np
import source.neural_network.__gradient__ as __gradient__
import source.neural_network.__gradient_descent__ as __gradient_descent__
import source.neural_network.__propagations__ as __propagations__
import source.neural_network.__cost__ as __cost__
import source.neural_network.__relu__ as __relu__
import source.neural_network.__sigmoid__ as __sigmoid__


last_cost = 0
#--
OPTIMIZATION_GRADIENT_DESCENT = 'gradient_descent'
OPTIMIZATION_BATCH_GRADIENT_DESCENT = 'batch_gradient_descent'
OPTIMIZATION_STOCHASTIC_GRADIENT_DESCENT ='stochastic_gradient_descent'
#--
ACTIVATION_RELU = 'relu'
ACTIVATION_SIGMOID = 'sigmoid'
ACTIVATION_TANH = 'tanh'
#--
METHOD_FIX_OVERFIT_REGULARIZATION = 'regularization'
METHOD_FIX_OVERFIT_DROPOUT = 'dropout'
"""
Here we use neural network with 2 hidden layers

Images we use: 20x20 (it's possible to change later)
"""
def start_nn_model_learning(X, Y, nn_architecture:list, optimization_algorithm_name,
                            activation_function_name, num_iterations,
                            learning_rate,
                            print_cost_function=None):
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
    nn_architecture = [INPUT_LAYER_SIZE] + nn_architecture + [NUMBER_OF_LABELS]
    # Initialize parameters
    parameters =  __gradient__.initialize_parameters_deep(nn_architecture)
    ## set up optimization algorithm for better learning
    optimization_algorithm = __gradient_descent__
    if (optimization_algorithm_name == OPTIMIZATION_GRADIENT_DESCENT):
        optimization_algorithm = __gradient_descent__
    elif (optimization_algorithm_name == OPTIMIZATION_STOCHASTIC_GRADIENT_DESCENT):
        raise BaseException('Stochastic gradient descent is not implemented yet!')
    elif (optimization_algorithm_name == OPTIMIZATION_BATCH_GRADIENT_DESCENT):
        raise BaseException('Batch gradient descent is not implemented yet!')
    ## set up activation function
    activation_module = None
    if (activation_function_name == ACTIVATION_TANH):
        activation_module = np.tanh
    elif (activation_function_name == ACTIVATION_SIGMOID):
        activation_module = __sigmoid__
    elif (activation_function_name == ACTIVATION_RELU):
        activation_module = __relu__

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        AL, caches = __propagations__.L_model_forward(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        current_cost = __cost__.compute_cost(AL, Y)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = __propagations__.L_model_backward(AL, Y, caches)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = optimization_algorithm.update_parameters(parameters, grads, learning_rate)

        # Print the cost every 1000 iterations
        if print_cost_function is not None and i % 10 == 0:
            print(str(current_cost))
            #print_cost_function(current_cost)

    return parameters

def preprocess_neural_network(dict_argument):
    start_nn_model_learning(X=dict_argument['X'],
                            Y=dict_argument['Y'],
                            optimization_algorithm_name=dict_argument['optimization_algorithm_name'],
                            nn_architecture=dict_argument['nn_architecture'],
                            activation_function_name=dict_argument['activation_function_name'],
                            num_iterations=dict_argument['num_iterations'],
                            learning_rate=dict_argument['learning_rate'],
                            print_cost_function=dict_argument['print_cost_function']
                            )

def get_last_cost():
    return last_cost