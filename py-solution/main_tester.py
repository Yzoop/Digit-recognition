import pytest
import numpy as np
import source.data_management.data_manager as data_manager
import source.neural_network.gradient as gradient


def test_training_data_downloader():
    assert 1 == 1
    assert data_manager.get_training_data() is not None


def test_gradient_koefs():
    test_width = 5
    test_height = 10
    testing_shape = (test_width, test_height)
    assert gradient.get_initialized_gradient(test_width, test_height).shape == testing_shape


def __is_any_value_huge__(np_gradient, bound) -> bool:
    for index, value in np.ndenumerate(np_gradient):
        if value > bound:
            return True
    return False


def test_gradient_big_values():
    test_width = 100
    test_height = 110
    max_value = 2

    my_theta = gradient.get_initialized_gradient(test_width, test_height)
    assert  __is_any_value_huge__(my_theta, max_value) == False