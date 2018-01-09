import sys
import scipy.io
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from visualise_images import print_examples
import param_plotting as pp

def main():
    # load the data
    X_full = scipy.io.loadmat('./matlabFiles/data28.mat')['data28'][0]
    X_full = np.array([x.reshape((784, )) for x in X_full])
    y_full = scipy.io.loadmat('./matlabFiles/labels28.mat')['labels28'].ravel() - 1

    if(option_set("knn")):
        knn_svd_pca(X_full, y_full)
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
        print("Accuracy of the model on training: " + str(model.score(X_train, y_train))
              + " and test: " + str(model.score(X_test, y_test)) + " data.")
    

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

def knn_svd_pca(X_full, y_full):
    for k_PCA in range(30, 100, 4):
        X_feat = svd_pca(X_full, k_PCA)

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.1, random_state=1)

        model = knn(X_train, y_train, X_test, y_test, k_PCA)


def knn(trainData, trainLabels, valData, valLabels, k_PCA):
    kVals = range(1, 30, 1)
    accuracies = []
 
    # loop over various values of `k` for the k-Nearest Neighbor classifier
    for k in kVals:
	    # train the k-Nearest Neighbor classifier with the current value of `k`
	    model = KNeighborsClassifier(n_neighbors=k)
	    model.fit(trainData, trainLabels)
 
	    # evaluate the model and update the accuracies list
	    score = model.score(valData, valLabels)
	    print("k_PCA=%d k_NN=%d, accuracy=%.2f%%" % (k_PCA, k, score * 100))
	    accuracies.append(score)
 
    # find the value of k that has the largest accuracy
    i = np.argmax(accuracies)
    print("k_PCA=%d k_NN=%d achieved highest accuracy of %.2f%% on validation data" % (k_PCA, kVals[i], accuracies[i] * 100))
    return model

def svd_pca(data, k):
    """Reduce DATA using its K principal components."""
    data = data.astype("float64")
    data -= np.mean(data, axis=0)
    U, S, V = np.linalg.svd(data, full_matrices=False)
    return U[:,:k].dot(np.diag(S)[:k,:k])

main()
