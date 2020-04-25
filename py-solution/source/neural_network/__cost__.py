import numpy as np
from source.data_management.data_manager import get_binary_matrix

def compute_cost(AL, Y):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    [Note that the parameters argument is not used in this function,
    but the auto-grader currently expects this parameter.
    Future version of this notebook will fix both the notebook
    and the auto-grader so that `parameters` is not needed.
    For now, please include `parameters` in the function signature,
    and also when invoking this function.]

    Returns:
    cost -- cross-entropy cost given equation (13)

    """

    m = Y.shape[1]  # number of example

    n = 10 #number of labels

    y_binary = get_binary_matrix(Y)

    # Compute the cross-entropy cost
    # sum_M = 0
    # for i in range(m):
    #     sum_K = 0
    #     for k in range(n):
    #         cur_y = 0
    #         if (Y[0, i] == k):
    #             cur_y = 1
    #         sum_K += -cur_y * np.log(A2[k, i]) - (1 - cur_y) * np.log(1 - A2[k, i])
    #     sum_M += sum_K
    #
    # cost = 1 / m * sum_M

    logprobs = np.multiply(-y_binary, np.log(AL)) - np.multiply((1 - y_binary), np.log(1 - AL))
    cost = 1 / m * np.sum(logprobs)

    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect.
    # E.g., turns [[17]] into 17

    return cost