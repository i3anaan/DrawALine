import time
from sklearn.linear_model import LogisticRegression


def testAccuracy(X_train, y_train, X_test, y_test, output_result):

    for C in range(-5, 6):
        time_start = time.time()
        model = LogisticRegression(max_iter=1000, C=3**C)
        time_training = time.time() - time_start
        model.fit(X_train, y_train)
        time_test = time.time() - time_training - time_start
        output_result(model, X_train, y_train, X_test, y_test)
