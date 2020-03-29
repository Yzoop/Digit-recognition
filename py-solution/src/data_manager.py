import scipy.io
#scipy for .mat file

# save name of a file with data
__mat_file_name__ = "data/digit_data.mat"

def get_training_data():
    """
    Function: get_training_data()

    Description: This function is used for reading and taking training data
                 (digit images)
    """
    training_data = scipy.io.loadmat(__mat_file_name__)

    return training_data
