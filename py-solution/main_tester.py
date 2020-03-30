import pytest
import numpy as np
import source.data_management.data_manager as data_manager
import source.neural_network.__propagations__ as propagations
import source.neural_network.__gradient__ as gradient
import source.__saved_test_data__ as saved_data

def test_gradient_koefs():
    test_n_x = 5
    test_n_h = 10
    test_n_y = 2
    grads = gradient.initial_parameters(test_n_x, test_n_h, test_n_y)
    assert grads is not None
    assert grads["W1"].shape == (test_n_h, test_n_x)
    assert grads["W2"].shape == (test_n_y, test_n_h)
    assert grads["b1"].shape == (test_n_h, 1)
    assert grads["b2"].shape == (test_n_y, 1)


def __is_any_value_huge__(np_gradient, bound) -> bool:
    for hidden_layer_id in ["W1", "W2"]:
        for index, value in np.ndenumerate(np_gradient[hidden_layer_id]):
            if abs(value) > bound:
                return True
    return False


def test_gradient_big_values():
    test_n_x = 61
    test_n_h = 120
    test_n_y = 10
    max_value = 0.50

    grads = gradient.initial_parameters(test_n_x, test_n_h, test_n_y)
    assert __is_any_value_huge__(grads, max_value) == False


def test_gradient_randomization():
    test_n_x = 61
    test_n_h = 120
    test_n_y = 10
    grads = gradient.initial_parameters(test_n_x, test_n_h, test_n_y)
    # determinant can be counted only for square matrixes
    for hidden_layer_id in ["W1", "W2"]:
        num_of_zeros = 0
        num_of_all_values = grads[hidden_layer_id].shape[0] * grads[hidden_layer_id].shape[1]
        for index, value in np.ndenumerate(grads[hidden_layer_id]):
            if value == 0:
                num_of_zeros += 1

    max_percent_of_zeros = 0.7 # we don't let gradient contain more than 70% of zeros
    assert num_of_zeros < num_of_all_values * max_percent_of_zeros



def test_init_params():
    np.random.seed(2)
    n_x = 4
    n_h = 5
    n_y = 3
    parameters = gradient.initial_parameters(n_x, n_h, n_y)
    test_parameters = saved_data.get_saved_parameters()

    assert (np.isclose(parameters['W1'], test_parameters['W1'])).all()
    assert (np.isclose(parameters['b1'], test_parameters['b1'])).all()
    assert (np.isclose(parameters['W2'], test_parameters['W2'])).all()
    assert (np.isclose(parameters['b2'], test_parameters['b2'])).all()


def test_forward_propagation():
    np.random.seed(2)
    n_x = 4
    m = 10 #num of examples
    X = np.random.randn(n_x, m)
    A2, cache = propagations.forward_propagation(X, saved_data.get_saved_parameters())
    saved_cache = saved_data.get_saved_cache()
    saved_a2 = saved_data.get_saved_a2()
    assert (np.isclose(A2, saved_a2)).all()
    assert (np.isclose(cache['Z1'], saved_cache['Z1'])).all()
    assert (np.isclose(cache['A1'], saved_cache['A1'])).all()
    assert (np.isclose(cache['Z2'], saved_cache['Z2'])).all()
    assert (np.isclose(cache['A2'], saved_cache['A2'])).all()
