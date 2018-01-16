import os
import os.path
import sys
import time
import scipy.io
import numpy as np

from sklearn.model_selection import train_test_split

import cls_knn
import cls_lda
import cls_qda
import cls_parzen
import cls_svc
import cls_mlp
import cls_log
import distortions

import csv


def main():
    # load the data
    full_path = os.path.realpath(__file__)
    X_full = scipy.io.loadmat(os.path.dirname(full_path) + '/../matlabFiles/data28.mat')['data28'][0]
    X_full = np.array([x.reshape((784,)) for x in X_full])
    y_full = scipy.io.loadmat(os.path.dirname(full_path) + '/../matlabFiles/labels28.mat')['labels28'].ravel() - 1

    # Split the data set
    if (option_set("--small")):
        print("Cherry picking data set...")
        X_train, X_test, y_train, y_test = cherry_pick_data_set(10, X_full, y_full)
    else:
        print("Splitting the data set...")
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.1, random_state=1)

    # Optionally extend the data set by using distortions
    if (option_set("--distort-shift")):
        print("Applying distortion...")
        X_train, y_train = distortions.extend_dataset_shift(X_train, y_train)
    if (option_set("--distort-grow")):
        print("Applying distortion...")
        X_train, y_train = distortions.extend_dataset_grow(X_train, y_train, 1)

    # print the shapes
    print("Training set size: " + str(X_train.shape))
    print("Test set size:     " + str(X_test.shape))

    if (option_set("lda")):
        cls_lda.testAccuracy(X_train, y_train, X_test, y_test, output_result)
    if (option_set("qda")):
        cls_qda.testAccuracy(X_train, y_train, X_test, y_test, output_result)
    if (option_set("parzen")):
        cls_parzen.testAccuracy(X_train, y_train, X_test, y_test, output_result)
    if (option_set("knn-pca")):
        cls_knn.knn_pca(X_train, y_train, X_test, y_test, output_result)
    if (option_set("knn")):
        cls_knn.testAccuracy(X_train, y_train, X_test, y_test, 0, output_result)
    if (option_set("svc")):
        cls_svc.testAccuracy(X_train, y_train, X_test, y_test, output_result)
    if (option_set("mlp")):
        cls_mlp.testAccuracy(X_train, y_train, X_test, y_test, output_result)
    if (option_set("log")):
        cls_log.testAccuracy(X_train, y_train, X_test, y_test, output_result)



def option_set(option):
    return (option in sys.argv)


def cherry_pick_data_set(amount, X_full, y_full):
    length = len(X_full)
    step = round(length / 10)
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(10):
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_full[step * i:(step * (i + 1))]
            , y_full[step * i:(step * (i + 1))]
            , test_size=(1 - (amount / step)), random_state=1)
        X_train.extend(X_train_temp)
        X_test.extend(X_test_temp)
        y_train.extend(y_train_temp)
        y_test.extend(y_test_temp)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def output_result(model, X_train, y_train, X_test, y_test, time_training=float("inf")):
    time_start = time.time()
    train_acc = model.score(X_train, y_train) * 100
    test_acc = model.score(X_test, y_test) * 100
    time_test = time.time() - time_start

    print("#>%s<#\nAccuracy:\n   Training: %.2d%% \n   Test: %.2d%% \nTime:\n   Training: %.4f\n   Test: %.4f\n\n" % (type(model).__name__, train_acc, test_acc, time_training, time_test))

    file_name = 'results_' + type(model).__name__ + '.csv'
    file_exists = os.path.isfile(file_name)

    with open(file_name, 'a') as csvfile:
        fieldnames = [
            'train_accuracy', 'test_accuracy', 'cls_name', 'train_shape', 'test_shape', 'time_training', 'time_test'
        ]
        fieldnames = fieldnames + list(model.get_params().keys())
        fieldnames.sort()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        data = {
            'train_accuracy': str(train_acc),
            'test_accuracy': str(test_acc),
            'cls_name': type(model).__name__,
            'train_shape': str(X_train.shape),
            'test_shape': str(X_test.shape),
            'time_training': str(time_training),
            'time_test': str(time_test),

        }
        data = {**data, **model.get_params()}

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow(data)


main()
