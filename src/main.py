import os
import os.path
import sys
import time
import datetime
import scipy.io
import numpy as np
import argparse

from sklearn.model_selection import train_test_split

import cls_knn
import cls_lda
import cls_qda
import cls_parzen
import cls_svc
import cls_mlp
import cls_log
import cls_qdc
import distortions
import similarity
from visualise_images import print_examples

import csv


def main():

    args = parse_arguments()

    scenario = 'small' if args.small else 'big'
    print(scenario)

    # load the data
    full_path = os.path.realpath(__file__)
    X_full = scipy.io.loadmat(os.path.dirname(full_path) + '/../matlabFiles/data28.mat')['data28'][0]
    X_full = np.array([x.reshape((784,)) for x in X_full])
    y_full = scipy.io.loadmat(os.path.dirname(full_path) + '/../matlabFiles/labels28.mat')['labels28'].ravel() - 1


    # Do an implementation test run on tiny data set
    if (args.test_run):
        X_full, X__, y_full, y__ = train_test_split(X_full, y_full, train_size=0.01, random_state=1)

    # Split the data set
    if (args.small and not args.test_run):
        print("Cherry picking data set...")
        X_train, X_test, y_train, y_test = cherry_pick_data_set(10, X_full, y_full)
    else:
        print("Splitting the data set...")
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.1, random_state=1)

    data_set = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }

    if args.distort is not None:
        # Extend the data set by using distortions
        if 'shift' in args.distort or 'all' in args.distort:
            print("Applying shift distortion...")
            X_train, y_train = distortions.extend_dataset_shift(X_train, y_train)
        if 'grow' in args.distort or 'all' in args.distort:
            print("Applying grow distortion...")
            X_train, y_train = distortions.extend_dataset_grow(X_train, y_train, 1)

    if args.similarity is not None:
        if 'dsim_edit' in args.distort:
            X_train, X_test = similarity.edit_distance_dissimilarity(X_train, X_test)
        if 'sim_norm1' in args.distort:
            X_train, X_test = similarity.norm_similarity(X_train, X_test, 1)
        if 'sim_norm2' in args.distort:
            X_train, X_test = similarity.norm_similarity(X_train, X_test, 2)
        if 'sim_cos' in args.distort:
            X_train, X_test = similarity.cosine_similarity_all(X_train, X_test)

    # print the shapes
    print("Training set size: " + str(X_train.shape))
    print("Test set size:     " + str(X_test.shape))

    if ('lda' in args.classifiers):
            run_batch(cls_lda, data_set, args)
    if ('qda' in args.classifiers):
        cls_qda.testAccuracy(X_train, y_train, X_test, y_test, output_result)
    if ('parzen' in args.classifiers):
        cls_parzen.testAccuracy(X_train, y_train, X_test, y_test, output_result)
    if ('knn-pca' in args.classifiers):
        cls_knn.knn_pca(X_train, y_train, X_test, y_test, output_result)
    if ('knn' in args.classifiers):
        cls_knn.testAccuracy(X_train, y_train, X_test, y_test, 0, output_result)
    if ('svc' in args.classifiers):
        cls_svc.testAccuracy(X_train, y_train, X_test, y_test, output_result)
    if ('mlp' in args.classifiers):
            run_batch(cls_mlp, data_set, args)
    if ('log' in args.classifiers):
        cls_log.testAccuracy(X_train, y_train, X_test, y_test, output_result)
    if ('qdc' in args.classifiers):
        cls_qdc.qdc_pca(X_train, y_train, X_test, y_test, output_result)


def parse_arguments():
    parser = argparse.ArgumentParser(prog='DrawALine', description='Pattern Recognition tool for recognizing decimals from the NIST data set.')
    parser.add_argument('classifiers', help='The classifiers to run', nargs='+', choices=['lda', 'qda', 'parzen', 'knn', 'knn-pca', 'svc', 'mlp', 'log', 'qdc'])


    parser.add_argument('--test-run', help='Run in implementation test mode - use a tiny data set', action='store_true')
    parser.add_argument('--small', help='Use a small training set', action='store_true')
    parser.add_argument('--distort', help='Distort the data', action='store', choices=['shift', 'grow', 'all'])
    parser.add_argument('--similarity', help='Transform the data to similarity representation', action='store', choices=['dsim_edit', 'sim_norm1', 'sim_norm2', 'sim_cos'])

    args = parser.parse_args()

    return args

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


def run_batch(cls, data_set, args):
    for model in cls.get_models('small' if args.small else 'big'):
        run_classifier(model, data_set, args)


def run_classifier(model, data_set, args):

    # Train
    time_start = time.time()
    model.fit(data_set['X_train'], data_set['y_train'])
    time_train = time.time() - time_start

    # Test
    time_start = time.time()
    train_acc = model.score(data_set['X_train'], data_set['y_train']) * 100.0
    time_test_train = time.time() - time_start
    test_acc = model.score(data_set['X_test'], data_set['y_test']) * 100.0
    time_test_test = time.time() - time_start - time_test_train

    results = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'cls_name': type(model).__name__,
        'train_shape': str(data_set['X_train'].shape),
        'test_shape': str(data_set['X_test'].shape),
        'time_training': time_train,
        'time_test_train': time_test_train,
        'time_test_test': time_test_test,
        'log_time': datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),

    }
    # results = {**results, **extra_params}
    results = {**results, **model.get_params()}

    print_results(model, data_set, results)
    log_results(model, data_set, results)


def print_results(model, data_set, results):
    print("#>%s<# Train/Test: %06.2f%%/%06.2f%%  Train/Test: %.04f/%.04f" % (type(model).__name__, results['train_accuracy'], results['test_accuracy'], results['time_training'], results['time_test_test']))


def log_results(model, data_set, results):
    file_name = 'results/results_' + type(model).__name__ + '.csv'
    file_exists = os.path.isfile(file_name)

    with open(file_name, 'a') as csvfile:
        fieldnames = list(results.keys())
        fieldnames.sort()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow(results)

main()
