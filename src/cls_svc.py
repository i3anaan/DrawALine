import time
from sklearn.svm import SVC


def testAccuracy(X_train, y_train, X_test, y_test, output_result):

    for kernel in ['rbf', 'linear', 'poly', 'sigmoid']:
        time_start = time.time()
        model = SVC(kernel=kernel)
        model.fit(X_train, y_train)
        time_training = time.time() - time_start
        output_result(model, X_train, y_train, X_test, y_test, time_training)

    for gamma in range(-3, 2):
        for C in range(-5, 6):
            time_start = time.time()
            model = SVC(C=3**C, gamma=10**gamma)
            model.fit(X_train, y_train)
            time_training = time.time() - time_start
            output_result(model, X_train, y_train, X_test, y_test, time_training)
