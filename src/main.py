import sys
import scipy.io
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

import cls_knn

from visualise_images import print_examples
import param_plotting as pp


def main():
    # load the data
    X_full = scipy.io.loadmat('../matlabFiles/data28.mat')['data28'][0]
    X_full = np.array([x.reshape((784, )) for x in X_full])
    y_full = scipy.io.loadmat(
        '../matlabFiles/labels28.mat')['labels28'].ravel() - 1

    if (option_set("knn-pca")):
        cls_knn.knn_svd_pca(X_full, y_full)
    if(option_set("knn-small")):
        X_small, y_small = get_some_data(10, X_full, y_full)
        X_train, X_test, y_train, y_test = train_test_split(
            X_small, y_small, test_size=0.1, random_state=1)
        model = cls_knn.knn(X_train, y_train, X_test, y_test, 0)
    else:

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.1, random_state=1)

        # print the shapes
        print("Training set size: " + str(X_train.shape))
        print("Test set size:     " + str(X_test.shape))

        # build the model
        #model = SVC(C=3)  # 10
        model = LogisticRegression(max_iter=1000, C=10)  # 10
        #model_reg = (lambda r: LogisticRegression(max_iter=1000, C=r))
        #model = MLPClassifier(
        #    solver='adam',
        #    alpha=0.01,  # 0.01
        #    hidden_layer_sizes=(800, 200, 30), # 800, 200, 30 -> 97.6
        #    random_state=1,
        #    max_iter=10000)

        model.fit(X_train, y_train)
        print("Accuracy of the model on training: " +
              str(model.score(X_train, y_train)) + " and test: " +
              str(model.score(X_test, y_test)) + " data.")

    if (option_set("--examples")):
        print_examples(model, X_test, y_test)


#model = SVC(C=3)  # 10
#model_reg = (lambda r: SVC(C=r))  # 10
#model = MLPClassifier(
#    solver='adam',
#    alpha=0.03,
#    hidden_layer_sizes=(800, 10),
#    random_state=1,
#    max_iter=10000)
#model_reg = (lambda r: MLPClassifier(solver='adam', alpha=r, hidden_layer_sizes=(800, 10), random_state=1, max_iter=10000))


def option_set(option):
    return (option in sys.argv)

def get_some_data(amount, X_full, y_full):
    length = len(X_full)
    step = round(length / 10)
    X_small = []
    y_small = []
    #X_small.append(X_full[0:length:step])
    for i in range(10):
        X_small.extend(X_full[step*i:(step*i + amount)])
        y_small.extend(y_full[step*i:(step*i + amount)])
    return X_small, y_small


main()
