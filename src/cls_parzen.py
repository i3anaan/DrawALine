import time
from sklearn.neighbors import KernelDensity


def testAccuracy(X_train, y_train, X_test, y_test, output_result):

    for kernel in ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']:
        time_start = time.time()
        model = KernelDensity(kernel=kernel)
        time_training = time.time() - time_start
        model.fit(X_train, y_train)
        time_test = time.time() - time_training - time_start
        output_result(model, X_train, y_train, X_test, y_test, time_training, time_test)
