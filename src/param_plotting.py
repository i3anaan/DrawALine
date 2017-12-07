"""
Functions to visualise the quality of the model with respect to its hyperparameters,
these hyperparameters are optimized with respect to the cross-validation set.
"""

import matplotlib.pyplot as plt

# default values to check for regularization
reg_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]

def make_graph(arr):
    """
    Takes an array of lines and plots these lines.

    :param arr - [([double],[double])] each tuple represenging (x,y) points in the graph
    :return A matplotlib object which contains the lines
    """

    plt.figure(1)
    for line in arr:
        (x, y) = line
        plt.plot(x, y)
    return plt

def training_size_classification_error(model, sizes, training_set, test_set):
    """
    Trains the model on subsets of the training data.

    :param model - The model to be trained
    :param sizes - The sizes of the training set to be evaluated
    :param training_set - (X,y) the training set
    :param test_set - (X,y) the test set
    :return [([double],[double])] each tuple represenging (x,y) the performance of the model
        on the set of the given size
    """

    training_errors = []
    test_errors = []
    (X_tr, y_tr) = training_set
    (X_te, y_te) = test_set

    for size in sizes:
        model.fit(X_tr[1:size], y_tr[1:size])
        training_errors.append(1 - model.score(X_tr[1:size], y_tr[1:size]))
        test_errors.append(1 - model.score(X_te, y_te))

    return [(sizes, training_errors), (sizes, test_errors)]


def regularization_classification_error(model_reg, regs, training_set,
                                        test_set):
    """
    Trains the model with various values of the regularization parameter

    :param model_reg - Function of type Double -> Model
    :param regs - The regularization parameter values to be evaluated
    :param traning_set - (X,y) the training set
    :param test_set - (X,y) the test set
    :return [([double],[double])] each tuple represenging (x,y) the performance of the model
        given the regularization parameter settings
    """

    training_errors = []
    test_errors = []
    (X_tr, y_tr) = training_set
    (X_te, y_te) = test_set

    for reg in regs:
        model = model_reg(reg)
        model.fit(X_tr, y_tr)
        training_errors.append(1 - model.score(X_tr, y_tr))
        test_errors.append(1 - model.score(X_te, y_te))

    return [(regs, training_errors), (regs, test_errors)]
