import source.data_management.data_manager as data_manager
import source.neural_network.gradient as gradient

"""
Here we use neural network with 2 hidden layers

Images we use: 20x20 (it's possible to change later)
"""

NUMBER_OF_LABELS = 10 # from 0 to 9
HIDDEN_LAYER_SIZE = 25
NUM_OF_PIXELS = 400 #as 20 * 20 = 400

if __name__ == "__main__":
    # start program with reading training data
    training_data = data_manager.get_local_training_data()

    # initialize by random small values gradient
    Theta1 = gradient.get_initialized_gradient(HIDDEN_LAYER_SIZE, NUM_OF_PIXELS + 1) # plus 1 bias unit
    Theta2 = gradient.get_initialized_gradient(NUMBER_OF_LABELS, HIDDEN_LAYER_SIZE + 1)

    