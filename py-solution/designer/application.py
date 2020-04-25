import sys
from source.neural_network import neural_network_model
from PyQt5 import QtWidgets, QtCore
from designer import main_window_design
from source.application_source.my_qt_thread_worker import ThreadWorker
from source.application_source import file_preprocess

class DigitRecognitionApp(QtWidgets.QMainWindow, main_window_design.Ui_MainWindow, QtCore.QRunnable):
    threadpool = None
    training_data = None
    OPTIMIZATION_ALGORITHM = 'optimization_algorithm'
    NAME = 'name'
    # ------------------------ neural network param keys ------------------------#
    GRADIENT_LEARNING_RATE = 'learning_rate'
    NEURAL_NETWORK_ARCHITECTURE = 'neural_network_architecture'
    ACTIVATION_FUNCTION = 'activation_function'
    OVERFITTING_ASSISTANT = 'overfitting_assistant'
    REGULARIZATION_PARAM = 'lambda'
    # ---------------------------------------------------------------------------#
    neural_network_params = {
        OPTIMIZATION_ALGORITHM: {NAME: None, GRADIENT_LEARNING_RATE: None},
        NEURAL_NETWORK_ARCHITECTURE: [],
        ACTIVATION_FUNCTION: None,
        OVERFITTING_ASSISTANT: {NAME: None, REGULARIZATION_PARAM: None}
    }
    __cost_graphic__ = []

    def __init__(self):
        super().__init__()
        # initialize the application
        self.setupUi(self)
        self.button_upload.clicked.connect(self.load_file)
        # optimization section
        self.radio_button_gradient_descent.toggled.connect(self.set_up_optimization_algorithm)
        self.radio_buttion_batch_gradient_descent.toggled.connect(self.set_up_optimization_algorithm)
        self.radio_button_stochastic_gradient_descent.toggled.connect(self.set_up_optimization_algorithm)
        self.button_apply_gradient_hyper_param.clicked.connect(self.set_gradient_hyper_param)
        # neural network acrhitecture section
        self.button_accept_num_of_hidden_layers.clicked.connect(self.set_num_of_hidden_layers)
        # activation function architecture
        self.radio_button_relu.toggled.connect(self.set_up_activation_function)
        self.radio_button_sigmoid.toggled.connect(self.set_up_activation_function)
        self.radio_button_tanh.toggled.connect(self.set_up_activation_function)
        # prevent overfitting
        self.radio_button_regularization.toggled.connect(self.set_up_overfit_avoider)
        self.button_apply_reg_param.clicked.connect(self.set_up_regularization_parameter)
        self.radio_button_dropout.toggled.connect(self.set_up_overfit_avoider)
        # final - start learning
        self.button_start_learning.clicked.connect(self.start_learning)
        # set up thread field
        self.threadpool = QtCore.QThreadPool()

    def set_up_regularization_parameter(self):
        self.neural_network_params[self.OVERFITTING_ASSISTANT][self.REGULARIZATION_PARAM] = \
            float(self.textedit_reg_param.text())

    def set_up_neural_network_from_table(self):
        network_size = len(self.neural_network_params[self.NEURAL_NETWORK_ARCHITECTURE])
        network_from_table = [int(self.table_neuron_architecture.item(0, c).text()) for c in range(network_size)]
        self.neural_network_params[self.NEURAL_NETWORK_ARCHITECTURE] = network_from_table

    def print_to_console(self, cost):
        self.label_say_here_loss_function.setText(str(cost))

    def start_learning(self):
        self.set_up_neural_network_from_table()
        print(self.neural_network_params)
        args = {'X' : self.training_data['X'],
                'Y':self.training_data['y'],
                'nn_architecture': self.neural_network_params[self.NEURAL_NETWORK_ARCHITECTURE],
                'optimization_algorithm_name':self.neural_network_params[self.OPTIMIZATION_ALGORITHM],
                'activation_function_name':self.neural_network_params[self.ACTIVATION_FUNCTION],
                'num_iterations': 800,
                'learning_rate': self.neural_network_params[self.OPTIMIZATION_ALGORITHM][self.GRADIENT_LEARNING_RATE],
                'print_cost_function':self.print_to_console}
        nn_thread = ThreadWorker(neural_network_model.preprocess_neural_network, args)
        nn_thread.set_function_on_finish(self.print_to_console)
        # start a thread for loading a training file (files can be very huge)
        self.threadpool.start(nn_thread)


    def set_up_overfit_avoider(self):
        radio_button_sender = self.sender()
        if radio_button_sender.isChecked():
            if radio_button_sender is self.radio_button_regularization:
                self.neural_network_params[self.OVERFITTING_ASSISTANT][self.NAME] = \
                    neural_network_model.METHOD_FIX_OVERFIT_REGULARIZATION
            if radio_button_sender is self.radio_button_dropout:
                self.neural_network_params[self.OVERFITTING_ASSISTANT][self.NAME] = \
                    neural_network_model.METHOD_FIX_OVERFIT_DROPOUT

    def set_up_activation_function(self):
        radio_button_sender = self.sender()
        if radio_button_sender.isChecked():
            if radio_button_sender is self.radio_button_sigmoid:
                self.neural_network_params[self.ACTIVATION_FUNCTION] = neural_network_model.ACTIVATION_SIGMOID
            elif radio_button_sender is self.radio_button_tanh:
                self.neural_network_params[self.ACTIVATION_FUNCTION] = neural_network_model.ACTIVATION_TANH
            elif radio_button_sender is self.radio_button_relu:
                self.neural_network_params[self.ACTIVATION_FUNCTION] = neural_network_model.ACTIVATION_RELU

    def set_num_of_hidden_layers(self):
        num_of_hidden_layers = int(self.number_of_hidden_layers.text())
        self.neural_network_params[self.NEURAL_NETWORK_ARCHITECTURE] = [0 for _ in range(num_of_hidden_layers)]
        self.label_status_num_of_hidden_layer.setText('Принято')
        self.table_neuron_architecture.setColumnCount(num_of_hidden_layers)
        self.table_neuron_architecture.setRowCount(1)
        self.table_neuron_architecture.setHorizontalHeaderLabels(
            ['Layer #' + str(i + 1) for i in range(num_of_hidden_layers)])

    def set_gradient_hyper_param(self):
        self.neural_network_params[self.OPTIMIZATION_ALGORITHM][self.GRADIENT_LEARNING_RATE] = \
            float(self.line_edit_descent_param.text())
        self.label_status_gradient_param.setText('Принято')

    def set_up_optimization_algorithm(self):
        radio_button_sender = self.sender()
        if radio_button_sender.isChecked():
            if radio_button_sender is self.radio_button_gradient_descent:
                self.neural_network_params[self.OPTIMIZATION_ALGORITHM][self.NAME] = \
                    neural_network_model.OPTIMIZATION_GRADIENT_DESCENT
            elif radio_button_sender is self.radio_buttion_batch_gradient_descent:
                self.neural_network_params[self.OPTIMIZATION_ALGORITHM][self.NAME] = \
                    neural_network_model.OPTIMIZATION_BATCH_GRADIENT_DESCENT
            elif radio_button_sender is self.radio_button_stochastic_gradient_descent:
                self.neural_network_params[self.OPTIMIZATION_ALGORITHM][self.NAME] = \
                    neural_network_model.OPTIMIZATION_STOCHASTIC_GRADIENT_DESCENT
            print('current = ', self.neural_network_params[self.OPTIMIZATION_ALGORITHM][self.NAME])

    def preview_file(self, data):
        num_of_rows = 20
        num_of_columns = 20
        self.table_preview_file.setRowCount(num_of_rows)
        self.table_preview_file.setColumnCount(num_of_columns)
        print('uploaded')
        # for row in range(num_of_rows):
        #     items = data['X'][row]
        #     for i in range(len(items)):
        #         self.table_preview_file.setItem(row, i, QtWidgets.QTableWidgetItem(items[i]))

    def on_load_file(self, data):
        if data is not None:
            self.training_data = data
            self.preview_file(data)
        else:
            print('can not load file')
        ##

    def load_file(self):
        # initialize a thread for file reading
        preprocess_data = ThreadWorker(file_preprocess.get_data_from_file)
        # run on_load_file function on success thread end
        preprocess_data.set_function_on_finish(self.on_load_file)
        # start a thread for loading a training file (files can be very huge)
        self.threadpool.start(preprocess_data)


def load_application():
    application = QtWidgets.QApplication(sys.argv)  # create an example of a QApplication
    window = DigitRecognitionApp()  # create an object of DigitRecognitionApp
    window.show()
    application.exec_()  # run DigitRecognitionApplication


if __name__ == "__main__":
    load_application()
