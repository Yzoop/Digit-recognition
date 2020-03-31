import source.data_management.data_manager as data_manager
import source.neural_network.neural_network_model as neural_network_model
import telegram
my_bot_api = '789233533:AAEbAUNs6bVfenwByVWcyRn4HGh-knlTzVs'


def start_main(num_iter=800):
    # start program with reading training data
    training_data = data_manager.get_training_data()  # first column is Y and other columns are X data (GRAY SCALE)

    # initialize by random small values gradient
    print("Please, wait: the neural network is being learned")
    parameters = neural_network_model.nn_model(X=training_data['X'],
                                               Y=training_data['y'],
                                               n_h=25,
                                               num_iterations=num_iter,
                                               print_cost=False)
    #print("Successfully learned!")


#if __name__ == "__main__":
bot = telegram.Bot(my_bot_api)

if bot.get_updates():
    chat_id = bot.get_updates()[-1].message.chat_id
    num_iters = 10000
    bot.send_message(chat_id, "Started telegram data science. Neural network started with " + str(num_iters))
    start_main(num_iters)
    bot.send_message(chat_id, "Finished!!!")
    bot.send_message(chat_id, "Last cost: " + str(neural_network_model.last_cost))

