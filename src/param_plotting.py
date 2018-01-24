"""
Functions to visualise the quality of the model with respect to its hyperparameters,
these hyperparameters are optimized with respect to the cross-validation set.
"""

import matplotlib.pyplot as plt
import csv

# default values to check for regularization
reg_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]

def make_graph(arr, labels):
    """
    Takes an array of lines and plots these lines.

    :param arr - [([double],[double])] each tuple represenging (x,y) points in the graph
    :return A matplotlib object which contains the lines
    """

    plt.figure(1)
    i = 0
    for line in arr:
        (x, y) = line
        plt.plot(x, y, linestyle='--', marker='o', label = labels[i])
        i= i+1
    return plt

def plot_from_file():
    cls = 'KNeighborsClassifier'
    labels = ['uniform', 'distance']
    #oneDict = {'weights': 'uniform', 'test_shape': '(1000, 30)' } 
    #secDict = {'weights': 'distance', 'test_shape': '(1000, 30)' } 
    allDict = []
    rowDict = {'test_shape': '(1000, 30)' }
    for lab in labels:
        dict = {'weights': lab, 'test_shape': '(1000, 30)' } 
        allDict.append(dict)
    xAxis = 'n_neighbors'
    yAxis = 'test_accuracy'
    #plots2 = make_plot_cls(cls, xAxis, yAxis, allDict)
    #plt = make_graph(plots, labels)
    plt = make_plot_cls_group_by_row(cls, xAxis, yAxis, 'weights', rowDict)
    plt.ylabel(yAxis)
    plt.xlabel(xAxis)
    plt.title(cls)
    #plt.yscale('logit')
    plt.grid(True)
    plt.legend()
    plt.show()

def make_plot_cls(model, xCol, yCol, rowDictValues):
    """
    :xCol - column of values for x Axis
    :yCol - column of values for y Axis
    Takes :rowDistValues as dictionary of column names and values of them that need to be satisfied

    """
    file_name = 'results_' + model + '.csv' #type(model).__name__
        #headers = d_reader.fieldnames
        #print(headers)
    plots= []
    for dict in rowDictValues:
        print(dict)
        with open(file_name, 'r') as f:
            d_reader = csv.DictReader(f)
           
            for row in d_reader:
                for key,value in dict.items():
                    add = True
                    if row[key] != value:
                        add = False
                        break
                if add:
                    #rows.append((num(row[xCol]), num(row[yCol])))
                    xRows.append(num(row[xCol]))
                    yRows.append(num(row[yCol]))
            plots.append((xRows, yRows))
            #plots.append(rows)
    return plots

def make_plot_cls_group_by_row(model, xCol, yCol, label, rowValues):
    """
    :xCol - column of values for x Axis
    :yCol - column of values for y Axis
    Groups data by :label
    :rowValues is a single directory that takes only those rows, that have a value specified in directory

    """
    file_name = 'results_' + model + '.csv' #type(model).__name__
    plots= []
    with open(file_name, 'r') as f:
        d_reader = csv.DictReader(f)
        #headers = d_reader.fieldnames
        #index = headers.index(label)
        #print(headers, index)
        rows = {}; xRows = {}; yRows = {};
        for row in d_reader:
            for key,value in rowValues.items():
                add = True
                if row[key] != value:
                    add = False
                    break
            if add:
                if not row[label] in xRows:
                    rows[row[label]] = []
                    xRows[row[label]] = []
                    yRows[row[label]] = []
                rows[row[label]].append((num(row[xCol]), num(row[yCol])))
                xRows[row[label]].append(num(row[xCol]))
                yRows[row[label]].append(num(row[yCol]))
        labels = []
        print(xRows)
        for k in xRows:
            plots.append((xRows[k], yRows[k]))
            labels.append(k)
            print(xRows[k], k)
    return make_graph(plots, labels)

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

plot_from_file()

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
