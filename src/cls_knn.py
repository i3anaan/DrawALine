import sys
import numpy as np
import math

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


#best values:
#k_NN = 3; k_PCA = 40 and 41 -> 97.80%
#k_NN = 1; k_PCA = 43 and 44 -> 97.80%
#without PCA: K_NN = 1 -> 96.00% very slow!
def knn_pca(X_train, y_train, X_test, y_test, output_result):
    for k_PCA in range(30, 51, 1):
        X_feat_train, X_feat_test = pca(X_train, X_test, k_PCA)

        model = testAccuracy(X_feat_train, y_train, X_feat_test, y_test, k_PCA,
                             output_result)


def testAccuracy(trainData, trainLabels, valData, valLabels, k_PCA,
                 output_result):
    if(len(trainLabels)>1000):
        kVals = range(1, 21, 1)
    else:
        kVals = range(1, 6, 1)
    accuracies = []

    weights = ['uniform', 'distance'];

    # loop over various values of `k` for the k-Nearest Neighbor classifier
    for w in weights:
        for k in kVals:
            # train the k-Nearest Neighbor classifier with the current value of `k`
            model = KNeighborsClassifier(n_neighbors=k, weights=w)
            model.fit(trainData, trainLabels)

            # evaluate the model and update the accuracies list
            score = model.score(valData, valLabels)
            if (k_PCA > 0):
                output_result(model, trainData, trainLabels, valData, valLabels)
                print("k_PCA=%d k_NN=%d weight=%s accuracy=%.2f%%" % (k_PCA, k, w, score * 100))
            else:
                output_result(model, trainData, trainLabels, valData, valLabels)
                print("k_NN=%d weight=%s accuracy=%.2f%%"  % (k, w, score * 100))
            accuracies.append(score)

    # find the value of k that has the largest accuracy
    i = np.argmax(accuracies)
    print(
        "k_PCA=%d k_NN=%d weight=%s achieved highest accuracy of %.2f%% on validation data"
        %(k_PCA, kVals[i%len(kVals)], weights[math.ceil(i/len(kVals))],  accuracies[i] * 100))
    return model


def pca(X_train, X_test, k):
    pca = PCA(n_components = k)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    return X_train, X_test
    #"""Reduce DATA using its K principal components."""
    #data = data.astype("float64")
    #data -= np.mean(data, axis=0)
    #U, S, V = np.linalg.svd(data, full_matrices=False)
    #return U[:, :k].dot(np.diag(S)[:k, :k])
