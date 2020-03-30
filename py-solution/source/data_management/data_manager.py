from urllib import request
from numpy import genfromtxt
import csv

# save name of a file with data
TRAINING_DATA_LINK = 'https://raw.githubusercontent.com/Yzoop/Digit-recognition/master/py-solution/source/data_management/training_data.csv'
TRAINING_DATA_LOCAL = 'data_management/training_data.csv'


def get_training_data():
    """
    Function: get_training_data()

    Description: This function is used for reading and taking training data
                 (digit images)

    Returns: dictionary with X and y
    """
    response = request.urlopen(TRAINING_DATA_LINK)
    training_data = genfromtxt(response, delimiter=',')
    y = training_data[0, :]
    X = training_data[1:, :]
    training_data_dict = {'y' : y, 'X' : X}

    return training_data_dict


def get_local_training_data():
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
    local_training_data = genfromtxt(TRAINING_DATA_LOCAL, delimiter=',')
    y = local_training_data[0, :]
    X = local_training_data[1:, :]
    training_data_dict = {'y' : y, 'X' : X}

    return training_data_dict
