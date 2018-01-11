import sys
import scipy.io
import numpy as np

from sklearn.model_selection import train_test_split

import cls_knn
import cls_svc
import cls_mlp
import distortions

from visualise_images import print_examples


def main():
    # load the data
    X_full = scipy.io.loadmat('../matlabFiles/data28.mat')['data28'][0]
    X_full = np.array([x.reshape((784, )) for x in X_full])
    y_full = scipy.io.loadmat(
        '../matlabFiles/labels28.mat')['labels28'].ravel() - 1

    # Optionally extend the data set by using distortions
    if (option_set("--distort")):
        print("Applying distortion...")
        X_full, y_full = distortions.extend_dataset_shift(X_full, y_full)

    # Split the data set
    print("Splitting the data set...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.1, random_state=1)

    # print the shapes
    print("Training set size: " + str(X_train.shape))
    print("Test set size:     " + str(X_test.shape))

    if (option_set("knn")):
        cls_knn.knn_svd_pca(X_full, y_full)
    if (option_set("svc")):
        cls_svc.testAccuracy(X_train, y_train, X_test, y_test)
    if (option_set("mlp")):
        cls_mlp.testAccuracy(X_train, y_train, X_test, y_test)

    if (option_set("--examples")):
        print_examples(model, X_test, y_test)


def option_set(option):
    return (option in sys.argv)


main()
