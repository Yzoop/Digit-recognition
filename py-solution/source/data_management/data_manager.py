from scipy.io import loadmat
import os
#scipy for .mat file

# save name of a file with data
__mat_file_name__ = 'data_management/digit_data.mat'

def get_training_data():
    """
    Function: get_training_data()

    Description: This function is used for reading and taking training data
                 (digit images)
    """
    print(os.getcwd())
    training_data = loadmat(__mat_file_name__)

    return training_data
