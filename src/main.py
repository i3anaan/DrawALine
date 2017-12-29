import sys
import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from classifiers import cls_manager as clss
import distortions

# this may be redundant
matplotlib.rcParams['backend'] = "Qt4Agg"

# load the data
X_full = scipy.io.loadmat('./matlabFiles/data28.mat')['matImages'][0]
X_full = np.array([x.reshape((784, )) for x in X_full])
y_full = scipy.io.loadmat('./matlabFiles/labels28.mat')['matLabels'].ravel() - 1
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.1, random_state=1)

X_train, y_train = distortions.extend_dataset_shift(X_train, y_train)

# print the shapes
print("Training set size: " + str(X_train.shape))
print("Test set size:     " + str(X_test.shape))

# image processing
# img = X[0, :]
# img = np.reshape(img, (16, 16)).T
# print(img)
# plt.imshow(img, cmap='gray')
# plt.savefig('test.png')

# build the model

# logistic.fit(X_train, y_train)
# svm.fit(X_train, y_train)

# clf.fit(X_train, y_train)

# overall accuracy of the model
# print("Accuracy Logistic Regression: " + str(logistic.score(X_train, y_train)) + " - " + str(logistic.score(X_test, y_test)))
# print("Accuracy Support Vector Machine: " + str(svm_model.score(X_train, y_train)) + " - " + str(svm_model.score(X_test, y_test)))
# print("Accuracy Neural Network: " + str(clf.score(X_train, y_train)) + " - " + str(clf.score(X_cv, y_cv)))


def fit_cls():

    logistic.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    mlp.fit(X_train, y_train)


def visualize_img(img):
    """
    Turns the array into a string and makes it more readable.

    :param img - A 2D array containing a single image
    :return The original array as a string with all 0 -> " " and 1 -> "#"
    """

    image = np.array_str(img)
    image = image.replace('0', ' ').replace('1', '#')
    return image


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
        model = model_reg
        training_errors.append(1 - model.score(X_tr, y_tr))
        test_errors.append(1 - model.score(X_te, y_te))

    return [(regs, training_errors), (regs, test_errors)]


def print_examples(model, test_set, test_set_answers):
    """
    Visualize model predictions on some instances.

    :param model - fully trained model
    :param test_set - the test set X values
    :param test_set_answers - the test set y values
    :return null
    """

    print("some examples: ")
    for i in range(0, len(test_set)):
        userin = input("Continue? (Y/n):")
        if userin == "n":
            break
        img = test_set[i]
        print(visualize_img(img.reshape((28, 28))))
        img2 = distortions.grow([img])[0]
        print(visualize_img(img2.reshape((28, 28))))
        prediction = model.predict(np.array([img]))
        print("Predicted value: " + str(prediction[0]))
        print("True value:      " + str(test_set_answers[i]))


# print("Accuracy Support Vector Machine: " + str(svm_model.score(X_train, y_train)) + " - " + str(svm_model.score(X_test, y_test)))
# print("Accuracy Neural Network: " + str(clf.score(X_train, y_train)) + " - " + str(clf.score(X_cv, y_cv)))

def option_set(option):
    return (option in sys.argv)


if not (option_set("--no-examples")):
    cls = clss.create_default_logistic((X_train, y_train))
    print_examples(cls, X_test, y_test)

if not (option_set("--no-display")):
    # set_sizes = [10,50] + list(range(100, 1801, 50))
    # print(training_size_classification_error(logistic, set_sizes, (X_train, y_train), (X_test, y_test)))
    # make_graph(training_size_classification_error(logistic, set_sizes, (X_train, y_train), (X_test, y_test))).savefig("SVM_trainingsize_clerror.png")
    reg_values = [
        0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000
    ]
    # print(regularization_classification_error(logistic_reg, reg_values, (X_train, y_train), (X_test, y_test)))

    cls = clss.create_custom_logistic((X_train, y_train), dict(C=0.03))

    make_graph(
        regularization_classification_error(
            cls, reg_values, (X_train, y_train),
            (X_test, y_test))).savefig("MLP_regularization_clerror.png")
