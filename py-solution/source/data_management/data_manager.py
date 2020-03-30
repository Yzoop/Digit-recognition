from numpy import genfromtxt
import os
#scipy for .mat file

# save name of a file with data
TRAINING_DATA = 'data_management/training_data.csv'

def get_training_data():
    """
    Function: get_training_data()

    Description: This function is used for reading and taking training data
                 (digit images)
    """
    print(os.getcwd())
    training_data = genfromtxt(TRAINING_DATA,delimiter=',')

    return training_data
