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

    Returns: numpy matrix of RGB pictures
    """
    response = request.urlopen(TRAINING_DATA_LINK)
    training_data = genfromtxt(response, delimiter=',')

    return training_data


def get_local_training_data():
    """
    Function: get_local_training_data()

    Description: Does the same thing as get_training_data(), but for speeding up
                 reads data from local directory
    """
    local_training_data = genfromtxt(TRAINING_DATA_LOCAL, delimiter=',')
    return local_training_data
