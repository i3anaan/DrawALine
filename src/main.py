import sys
import scipy.io
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from visualise_images import print_examples
import param_plotting as pp

def main():
    # load the data
    X_full = scipy.io.loadmat('../matlabFiles/data28.mat')['matImages'][0]
    X_full = np.array([x.reshape((784, )) for x in X_full])
    y_full = scipy.io.loadmat('../matlabFiles/labels28.mat')['matLabels'].ravel() - 1
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.1, random_state=1)

    # print the shapes
    print("Training set size: " + str(X_train.shape))
    print("Test set size:     " + str(X_test.shape))

    # build the model
    model = LogisticRegression(max_iter=1000, C=10)
    #model_reg = (lambda r: LogisticRegression(max_iter=1000, C=r))

    model.fit(X_train, y_train)
    print("Accuracy of the model on training: " + str(model.score(X_train, y_train))
          + " and test: " + str(model.score(X_test, y_test)) + " data.")

    if not (option_set("--no-examples")):
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

main()
