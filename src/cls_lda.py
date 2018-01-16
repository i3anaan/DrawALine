import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def testAccuracy(X_train, y_train, X_test, y_test, output_result):
    time_start = time.time()
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    time_training = time.time() - time_start
    output_result(model, X_train, y_train, X_test, y_test, time_training)
