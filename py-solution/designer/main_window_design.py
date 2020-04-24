# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(722, 1091)
        font = QtGui.QFont()
        font.setFamily("Corbel")
        font.setPointSize(12)
        MainWindow.setFont(font)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.tabWidget.setElideMode(QtCore.Qt.ElideNone)
        self.tabWidget.setObjectName("tabWidget")
        self.data_preprocessing = QtWidgets.QWidget()
        self.data_preprocessing.setObjectName("data_preprocessing")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.data_preprocessing)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.groupbox_choose_file = QtWidgets.QGroupBox(self.data_preprocessing)
        self.groupbox_choose_file.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.groupbox_choose_file.setObjectName("groupbox_choose_file")
        self.gridLayout = QtWidgets.QGridLayout(self.groupbox_choose_file)
        self.gridLayout.setObjectName("gridLayout")
        self.button_upload = QtWidgets.QPushButton(self.groupbox_choose_file)
        self.button_upload.setObjectName("button_upload")
        self.gridLayout.addWidget(self.button_upload, 0, 0, 1, 1)
        self.verticalLayout_8.addWidget(self.groupbox_choose_file)
        self.groupbox_preview = QtWidgets.QGroupBox(self.data_preprocessing)
        self.groupbox_preview.setObjectName("groupbox_preview")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.groupbox_preview)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.table_preview_file = QtWidgets.QTableWidget(self.groupbox_preview)
        self.table_preview_file.setObjectName("table_preview_file")
        self.table_preview_file.setColumnCount(0)
        self.table_preview_file.setRowCount(0)
        self.verticalLayout_9.addWidget(self.table_preview_file)
        self.verticalLayout_8.addWidget(self.groupbox_preview)
        self.pushButton = QtWidgets.QPushButton(self.data_preprocessing)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_8.addWidget(self.pushButton)
        self.tabWidget.addTab(self.data_preprocessing, "")
        self.tab_learning = QtWidgets.QWidget()
        self.tab_learning.setObjectName("tab_learning")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab_learning)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupbox_optimization_method = QtWidgets.QGroupBox(self.tab_learning)
        self.groupbox_optimization_method.setFlat(False)
        self.groupbox_optimization_method.setObjectName("groupbox_optimization_method")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupbox_optimization_method)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.radio_button_gradient_descent = QtWidgets.QRadioButton(self.groupbox_optimization_method)
        self.radio_button_gradient_descent.setAutoExclusive(True)
        self.radio_button_gradient_descent.setAutoRepeatDelay(0)
        self.radio_button_gradient_descent.setObjectName("radio_button_gradient_descent")
        self.verticalLayout_2.addWidget(self.radio_button_gradient_descent)
        self.radio_buttion_batch_gradient_descent = QtWidgets.QRadioButton(self.groupbox_optimization_method)
        self.radio_buttion_batch_gradient_descent.setAutoFillBackground(False)
        self.radio_buttion_batch_gradient_descent.setObjectName("radio_buttion_batch_gradient_descent")
        self.verticalLayout_2.addWidget(self.radio_buttion_batch_gradient_descent)
        self.radio_button_stochastic_gradient_descent = QtWidgets.QRadioButton(self.groupbox_optimization_method)
        self.radio_button_stochastic_gradient_descent.setObjectName("radio_button_stochastic_gradient_descent")
        self.verticalLayout_2.addWidget(self.radio_button_stochastic_gradient_descent)
        self.groupbox_descent_param = QtWidgets.QGroupBox(self.groupbox_optimization_method)
        self.groupbox_descent_param.setTitle("")
        self.groupbox_descent_param.setObjectName("groupbox_descent_param")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupbox_descent_param)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_descent_param = QtWidgets.QLabel(self.groupbox_descent_param)
        self.label_descent_param.setObjectName("label_descent_param")
        self.horizontalLayout_4.addWidget(self.label_descent_param)
        self.line_edit_descent_param = QtWidgets.QLineEdit(self.groupbox_descent_param)
        self.line_edit_descent_param.setText("")
        self.line_edit_descent_param.setObjectName("line_edit_descent_param")
        self.horizontalLayout_4.addWidget(self.line_edit_descent_param)
        self.button_apply_gradient_hyper_param = QtWidgets.QPushButton(self.groupbox_descent_param)
        self.button_apply_gradient_hyper_param.setObjectName("button_apply_gradient_hyper_param")
        self.horizontalLayout_4.addWidget(self.button_apply_gradient_hyper_param)
        self.label_status_gradient_param = QtWidgets.QLabel(self.groupbox_descent_param)
        self.label_status_gradient_param.setStyleSheet("QLabel {\n"
"        color: green;\n"
"    font: 12pt \"MS Shell Dlg 2\";\n"
"    }")
        self.label_status_gradient_param.setText("")
        self.label_status_gradient_param.setAlignment(QtCore.Qt.AlignCenter)
        self.label_status_gradient_param.setObjectName("label_status_gradient_param")
        self.horizontalLayout_4.addWidget(self.label_status_gradient_param)
        self.verticalLayout_2.addWidget(self.groupbox_descent_param)
        self.verticalLayout.addWidget(self.groupbox_optimization_method)
        self.groupbox_set_neural_network = QtWidgets.QGroupBox(self.tab_learning)
        self.groupbox_set_neural_network.setFlat(False)
        self.groupbox_set_neural_network.setObjectName("groupbox_set_neural_network")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupbox_set_neural_network)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupbox_hidden_layers = QtWidgets.QGroupBox(self.groupbox_set_neural_network)
        self.groupbox_hidden_layers.setTitle("")
        self.groupbox_hidden_layers.setObjectName("groupbox_hidden_layers")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.groupbox_hidden_layers)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label = QtWidgets.QLabel(self.groupbox_hidden_layers)
        self.label.setObjectName("label")
        self.horizontalLayout_6.addWidget(self.label)
        self.number_of_hidden_layers = QtWidgets.QLineEdit(self.groupbox_hidden_layers)
        self.number_of_hidden_layers.setObjectName("number_of_hidden_layers")
        self.horizontalLayout_6.addWidget(self.number_of_hidden_layers)
        self.button_accept_num_of_hidden_layers = QtWidgets.QPushButton(self.groupbox_hidden_layers)
        self.button_accept_num_of_hidden_layers.setObjectName("button_accept_num_of_hidden_layers")
        self.horizontalLayout_6.addWidget(self.button_accept_num_of_hidden_layers)
        self.label_status_num_of_hidden_layer = QtWidgets.QLabel(self.groupbox_hidden_layers)
        self.label_status_num_of_hidden_layer.setStyleSheet("QLabel {\n"
"        color: green;\n"
"    font: 12pt \"MS Shell Dlg 2\";\n"
"    }")
        self.label_status_num_of_hidden_layer.setText("")
        self.label_status_num_of_hidden_layer.setObjectName("label_status_num_of_hidden_layer")
        self.horizontalLayout_6.addWidget(self.label_status_num_of_hidden_layer)
        self.verticalLayout_3.addWidget(self.groupbox_hidden_layers)
        self.table_neuron_architecture = QtWidgets.QTableWidget(self.groupbox_set_neural_network)
        self.table_neuron_architecture.setObjectName("table_neuron_architecture")
        self.table_neuron_architecture.setColumnCount(0)
        self.table_neuron_architecture.setRowCount(0)
        self.verticalLayout_3.addWidget(self.table_neuron_architecture)
        self.verticalLayout.addWidget(self.groupbox_set_neural_network)
        self.groupbox_activation_function = QtWidgets.QGroupBox(self.tab_learning)
        self.groupbox_activation_function.setFlat(False)
        self.groupbox_activation_function.setObjectName("groupbox_activation_function")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupbox_activation_function)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.radio_button_relu = QtWidgets.QRadioButton(self.groupbox_activation_function)
        self.radio_button_relu.setObjectName("radio_button_relu")
        self.verticalLayout_4.addWidget(self.radio_button_relu)
        self.radio_button_sigmoid = QtWidgets.QRadioButton(self.groupbox_activation_function)
        self.radio_button_sigmoid.setObjectName("radio_button_sigmoid")
        self.verticalLayout_4.addWidget(self.radio_button_sigmoid)
        self.radio_button_tanh = QtWidgets.QRadioButton(self.groupbox_activation_function)
        self.radio_button_tanh.setObjectName("radio_button_tanh")
        self.verticalLayout_4.addWidget(self.radio_button_tanh)
        self.verticalLayout.addWidget(self.groupbox_activation_function)
        self.groupbox_variance_optimization = QtWidgets.QGroupBox(self.tab_learning)
        self.groupbox_variance_optimization.setObjectName("groupbox_variance_optimization")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupbox_variance_optimization)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.groupbox_regularization = QtWidgets.QGroupBox(self.groupbox_variance_optimization)
        self.groupbox_regularization.setTitle("")
        self.groupbox_regularization.setObjectName("groupbox_regularization")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupbox_regularization)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.radio_button_regularization = QtWidgets.QRadioButton(self.groupbox_regularization)
        self.radio_button_regularization.setObjectName("radio_button_regularization")
        self.horizontalLayout_2.addWidget(self.radio_button_regularization)
        self.line = QtWidgets.QFrame(self.groupbox_regularization)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_2.addWidget(self.line)
        self.label_ask_user_for_reg_param = QtWidgets.QLabel(self.groupbox_regularization)
        self.label_ask_user_for_reg_param.setObjectName("label_ask_user_for_reg_param")
        self.horizontalLayout_2.addWidget(self.label_ask_user_for_reg_param)
        self.groupbox_reg_param = QtWidgets.QGroupBox(self.groupbox_regularization)
        self.groupbox_reg_param.setTitle("")
        self.groupbox_reg_param.setObjectName("groupbox_reg_param")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupbox_reg_param)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.textedit_reg_param = QtWidgets.QLineEdit(self.groupbox_reg_param)
        self.textedit_reg_param.setObjectName("textedit_reg_param")
        self.horizontalLayout_3.addWidget(self.textedit_reg_param)
        self.button_apply_reg_param = QtWidgets.QPushButton(self.groupbox_reg_param)
        self.button_apply_reg_param.setObjectName("button_apply_reg_param")
        self.horizontalLayout_3.addWidget(self.button_apply_reg_param)
        self.horizontalLayout_2.addWidget(self.groupbox_reg_param)
        self.verticalLayout_5.addWidget(self.groupbox_regularization)
        self.groupbox_dropout = QtWidgets.QGroupBox(self.groupbox_variance_optimization)
        self.groupbox_dropout.setTitle("")
        self.groupbox_dropout.setObjectName("groupbox_dropout")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupbox_dropout)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.radio_button_dropout = QtWidgets.QRadioButton(self.groupbox_dropout)
        self.radio_button_dropout.setObjectName("radio_button_dropout")
        self.verticalLayout_6.addWidget(self.radio_button_dropout)
        self.verticalLayout_5.addWidget(self.groupbox_dropout)
        self.verticalLayout.addWidget(self.groupbox_variance_optimization)
        self.button_start_learning = QtWidgets.QPushButton(self.tab_learning)
        self.button_start_learning.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";\n"
"border-color: rgb(170, 85, 255);\n"
"color: rgb(0, 170, 0);")
        self.button_start_learning.setObjectName("button_start_learning")
        self.verticalLayout.addWidget(self.button_start_learning)
        self.graphicsView = QtWidgets.QGraphicsView(self.tab_learning)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout.addWidget(self.graphicsView)
        self.groupbox_current_loss_function = QtWidgets.QGroupBox(self.tab_learning)
        self.groupbox_current_loss_function.setTitle("")
        self.groupbox_current_loss_function.setObjectName("groupbox_current_loss_function")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.groupbox_current_loss_function)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_say_here_loss_function = QtWidgets.QLabel(self.groupbox_current_loss_function)
        self.label_say_here_loss_function.setObjectName("label_say_here_loss_function")
        self.horizontalLayout_5.addWidget(self.label_say_here_loss_function)
        self.label_current_loss = QtWidgets.QLabel(self.groupbox_current_loss_function)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_current_loss.setFont(font)
        self.label_current_loss.setObjectName("label_current_loss")
        self.horizontalLayout_5.addWidget(self.label_current_loss)
        self.verticalLayout.addWidget(self.groupbox_current_loss_function)
        self.tabWidget.addTab(self.tab_learning, "")
        self.tab_using = QtWidgets.QWidget()
        self.tab_using.setObjectName("tab_using")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.tab_using)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.groupBox = QtWidgets.QGroupBox(self.tab_using)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_7.addWidget(self.groupBox)
        self.tabWidget.addTab(self.tab_using, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Распознавание цифры на изображении"))
        self.groupbox_choose_file.setTitle(_translate("MainWindow", "Выбрать файл с данными для обучения"))
        self.button_upload.setText(_translate("MainWindow", "Загрузить..."))
        self.groupbox_preview.setTitle(_translate("MainWindow", "Предпросмотр"))
        self.pushButton.setText(_translate("MainWindow", "Сохранить"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.data_preprocessing), _translate("MainWindow", "Подготовка данных для обучения"))
        self.groupbox_optimization_method.setTitle(_translate("MainWindow", "Метод оптимизации"))
        self.radio_button_gradient_descent.setAccessibleName(_translate("MainWindow", "gradient_descent"))
        self.radio_button_gradient_descent.setText(_translate("MainWindow", "Градиентный спуск"))
        self.radio_buttion_batch_gradient_descent.setAccessibleName(_translate("MainWindow", "batch_gradient_descent"))
        self.radio_buttion_batch_gradient_descent.setText(_translate("MainWindow", "Пакетный градиентный спуск"))
        self.radio_button_stochastic_gradient_descent.setAccessibleName(_translate("MainWindow", "stochastic_gradient_descent"))
        self.radio_button_stochastic_gradient_descent.setText(_translate("MainWindow", "Стохастический градиентный спуск"))
        self.label_descent_param.setText(_translate("MainWindow", "Параметр спуска:"))
        self.button_apply_gradient_hyper_param.setText(_translate("MainWindow", "Принять"))
        self.groupbox_set_neural_network.setTitle(_translate("MainWindow", "Архитектура нейронной сети"))
        self.label.setText(_translate("MainWindow", "Количество скрытых слоев:"))
        self.button_accept_num_of_hidden_layers.setText(_translate("MainWindow", "Принять"))
        self.groupbox_activation_function.setTitle(_translate("MainWindow", "Функция активации скрытых нейронов"))
        self.radio_button_relu.setText(_translate("MainWindow", "ReLU"))
        self.radio_button_sigmoid.setText(_translate("MainWindow", "Sigmoid"))
        self.radio_button_tanh.setText(_translate("MainWindow", "Tanh"))
        self.groupbox_variance_optimization.setTitle(_translate("MainWindow", "Метод предотвращения переобучения"))
        self.radio_button_regularization.setText(_translate("MainWindow", "Регуляризация"))
        self.label_ask_user_for_reg_param.setText(_translate("MainWindow", "Параметр регуляризации:"))
        self.button_apply_reg_param.setText(_translate("MainWindow", "Принять"))
        self.radio_button_dropout.setText(_translate("MainWindow", "\"Выбивание\" нейрона"))
        self.button_start_learning.setText(_translate("MainWindow", "Начать обучение нейронной сети"))
        self.label_say_here_loss_function.setText(_translate("MainWindow", "Текущее значение функции потерь:"))
        self.label_current_loss.setText(_translate("MainWindow", "000"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_learning), _translate("MainWindow", "Обучение"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_using), _translate("MainWindow", "Использование"))