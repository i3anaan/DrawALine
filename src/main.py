import sys
import scipy.io
import numpy as np

from sklearn.model_selection import train_test_split

import cls_knn
import cls_svc
import cls_mlp
import distortions
from visualise_images import print_examples

import csv


def main():
    # load the data
    X_full = scipy.io.loadmat('../matlabFiles/data28.mat')['data28'][0]
    X_full = np.array([x.reshape((784, )) for x in X_full])
    y_full = scipy.io.loadmat(
        '../matlabFiles/labels28.mat')['labels28'].ravel() - 1

    print("Splitting the data set...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.1, random_state=1)

    # Split the data set
    if (option_set("--small")):
        print("Cherry picking data set...")
        X_train, y_train = cherry_pick_data_set(10, X_full, y_full)


    # Optionally extend the data set by using distortions
    if (option_set("--distort")):
        print("Applying distortion...")
        X_train, y_train = distortions.extend_dataset_shift(X_train, y_train)

    # print the shapes
    print("Training set size: " + str(X_train.shape))
    print("Test set size:     " + str(X_test.shape))

    if (option_set("knn-pca")):
        cls_knn.knn_pca(X_train, y_train, X_test, y_test, output_result)
    if (option_set("knn")):
        cls_knn.testAccuracy(X_train, y_train, X_test, y_test, 0, output_result)
    if (option_set("svc")):
        cls_svc.testAccuracy(X_train, y_train, X_test, y_test, output_result)
    if (option_set("mlp")):
        cls_mlp.testAccuracy(X_train, y_train, X_test, y_test, output_result)


def option_set(option):
    return (option in sys.argv)


def cherry_pick_data_set(amount, X_full, y_full):
    length = len(X_full)
    step = round(length / 10)
    X_small = []
    y_small = []
    for i in range(10):
        X_small.extend(X_full[step * i:((step * i) + amount)])
        y_small.extend(y_full[step * i:(step * i + amount)])
    return np.array(X_small), np.array(y_small)


def output_result(model, X_train, y_train, X_test, y_test):
    print("Accuracy of the model on training: " +
          str(model.score(X_train, y_train)) + " and test: " +
          str(model.score(X_test, y_test)) + " data.")

    with open('results.csv', 'w') as csvfile:
        fieldnames = ['train_accuracy', 'test_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({
            'train_accuracy': str(model.score(X_train, y_train)),
            'test_accuracy': str(model.score(X_test, y_test))
        })


main()
