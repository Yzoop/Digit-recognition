from urllib import request
import numpy as np
import random
import csv

# save name of a file with data
TRAINING_DATA_LINK = 'https://raw.githubusercontent.com/Yzoop/Digit-recognition/master/py-solution/source/data_management/training_data.csv'
TRAINING_DATA_LOCAL = 'data_management/training_data.csv'
NUM_OF_LABELS = 10


def get_training_data():
    """
    Function: get_training_data()

    Description: This function is used for reading and taking training data
                 (digit images)

    Returns: dictionary with X and y
    """
    response = request.urlopen(TRAINING_DATA_LINK)
    training_data = np.genfromtxt(response, delimiter=',')
    y = np.array(training_data[0, :], ndmin=2)
    for index, val in np.ndenumerate(y):
        y[index[0], index[1]] = 0 if y[index[0], index[1]] == 10 else y[index[0], index[1]]
    X = training_data[1:, :]
    training_data_dict = {'y': y, 'X': X}

    return training_data_dict


def get_local_training_data(path=None):
    """
    Function: get_local_training_data()

    Description: Does the same thing as get_training_data(), but for speeding up
                 reads data from local directory

                 First RAW - vector of labels. from 1 to 10.
                 10 - 0
                 1 - 1
                 2 - 2
                 3 - 3
                 4 - 4
                 5 - 5
                 6 - 6
                 7 - 7
                 8 - 8
                 9 - 9
    """
    local_training_data = np.genfromtxt(TRAINING_DATA_LOCAL if path is None else path, delimiter=',')
    y = np.array(local_training_data[0, :], ndmin=2)
    for index, val in np.ndenumerate(y):
        y[index[0], index[1]] = 0 if y[index[0], index[1]] == 10 else y[index[0], index[1]]
    X = local_training_data[1:, :]
    training_data_dict = {'y': y, 'X': X}

    return training_data_dict


def get_binary_matrix(Y, num_of_labels=NUM_OF_LABELS):
    assert Y.shape[0] == 1  # is vector
    binary_matrix = np.zeros((num_of_labels, Y.shape[1]))
    for index, value in np.ndenumerate(Y):
        i = index[0], index[1]
        binary_matrix[int(Y[0, index[1]]), index[1]] = 1.0
    return binary_matrix
