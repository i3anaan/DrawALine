import sys
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#best values:
#k_NN = 3; k_PCA = 40 and 41 -> 97.80%
#k_NN = 1; k_PCA = 43 and 44 -> 97.80%
#without PCA: K_NN = 1 -> 96.00% very slow!
def knn_svd_pca(X_full, y_full):
    for k_PCA in range(35, 45, 1):
        X_feat = svd_pca(X_full, k_PCA)

        X_train, X_test, y_train, y_test = train_test_split(
            X_feat, y_full, test_size=0.1, random_state=1)

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
    print(
        "k_PCA=%d k_NN=%d achieved highest accuracy of %.2f%% on validation data"
        % (k_PCA, kVals[i], accuracies[i] * 100))
    return model


def svd_pca(data, k):
    """Reduce DATA using its K principal components."""
    data = data.astype("float64")
    data -= np.mean(data, axis=0)
    U, S, V = np.linalg.svd(data, full_matrices=False)
    return U[:, :k].dot(np.diag(S)[:k, :k])
