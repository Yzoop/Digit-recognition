import source.data_management.data_manager as data_manager
import source.neural_network.neural_network_model as neural_network_model
my_bot_api = '789233533:AAEbAUNs6bVfenwByVWcyRn4HGh-knlTzVs'

def console_print_cost(cost):
    print('current cost: ', cost)


def start_main(num_iter=800):
    # start program with reading training data
    training_data = data_manager.get_local_training_data()  # first column is Y and other columns are X data (GRAY SCALE)

    # initialize by random small values gradient
    print("Please, wait: the neural network is being learned")
    parameters = neural_network_model.start_nn_model_learning(X=training_data['X'],
                                                              Y=training_data['y'],
                                                              n_h=25,
                                                              optimization_algorithm_name=neural_network_model.OPTIMIZATION_GRADIENT_DESCENT,
                                                              num_iterations=num_iter,
                                                              print_cost_function=console_print_cost)
    #print("Successfully learned!")


if __name__ == "__main__":
    num_iters = 500
    start_main(num_iters)

