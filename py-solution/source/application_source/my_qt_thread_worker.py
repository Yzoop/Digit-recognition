from PyQt5 import QtCore


class ThreadWorker(QtCore.QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    """
    __thread_answer__ = None
    __func_on_finish__ = None

    def __init__(self, fn, *args, **kwargs):
        super(ThreadWorker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def set_function_on_finish(self, function_on_finish):
        self.__func_on_finish__ = function_on_finish

    @QtCore.pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        self.__thread_answer__ = (self.fn(*self.args, **self.kwargs))
        self.__func_on_finish__(self.__thread_answer__)