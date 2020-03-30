import source.data_management.data_manager as data_manager
import source.neural_network.neural_network_model as neural_network_model


if __name__ == "__main__":
    # start program with reading training data
    training_data = data_manager.get_local_training_data() #first column is Y and other columns are X data (GRAY SCALE)

    # initialize by random small values gradient
    parameters = neural_network_model.nn_model(X=training_data['X'],
                                               Y=training_data['y'],
                                               n_h=25,
                                               num_iterations=800,
                                               print_cost=True)
