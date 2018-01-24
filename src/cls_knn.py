import sys
import numpy as np
import math
import time
import feat_pca

from sklearn.neighbors import KNeighborsClassifier


#best values:
#k_NN = 3; k_PCA = 40 and 41 -> 97.80%
#k_NN = 1; k_PCA = 43 and 44 -> 97.80%
#without PCA: K_NN = 1 -> 96.00% very slow!
def knn_pca(X_train, y_train, X_test, y_test, output_result):
    for k_PCA in range(37, 51, 1):
        X_feat_train, X_feat_test = feat_pca.pca(X_train, X_test, k_PCA)

        model = testAccuracy(X_feat_train, y_train, X_feat_test, y_test, k_PCA,
                             output_result)


def testAccuracy(X_train, y_train, X_test, y_test, k_PCA,
                 output_result):
    if(len(y_train)>1000):
        kVals = range(1, 21, 1)
    else:
        kVals = range(1, 6, 1)
    accuracies = []

    weights = ['uniform', 'distance'];

    # loop over various values of `k` for the k-Nearest Neighbor classifier
    for w in weights:
        for k in kVals:
            # train the k-Nearest Neighbor classifier with the current value of `k`
            time_start = time.time()
            model = KNeighborsClassifier(n_neighbors=k, weights=w)
            model.fit(X_train, y_train)
            time_training = time.time() - time_start
            # evaluate the model and update the accuracies list
            score = model.score(X_test, y_test)
            if (k_PCA > 0):
                print("k_PCA=%d k_NN=%d weight=%s accuracy=%.2f%%" % (k_PCA, k, w, score * 100))
            else:
                print("k_NN=%d weight=%s accuracy=%.2f%%"  % (k, w, score * 100))
                
            output_result(model, X_train, y_train, X_test, y_test, time_training)
            accuracies.append(score)

    # find the value of k that has the largest accuracy
    i = np.argmax(accuracies)
    print(
        "k_PCA=%d k_NN=%d weight=%s achieved highest accuracy of %.2f%% on validation data"
        %(k_PCA, kVals[i%len(kVals)], weights[math.floor(i/len(kVals))],  accuracies[i] * 100))
    return model
