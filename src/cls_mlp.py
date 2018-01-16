import time
from sklearn.neural_network import MLPClassifier


def testAccuracy(X_train, y_train, X_test, y_test, output_result):

    for activation in ['identity', 'logistic', 'tanh', 'relu']:
        for alpha in range(-5,5):
            time_start = time.time()
            model = MLPClassifier(
                alpha=3**alpha,  # 0.01
                hidden_layer_sizes=(800, 200, 30),  # 800, 200, 30 -> 97.6
                random_state=1,
                activation=activation,
                max_iter=10000)

            model.fit(X_train, y_train)
            time_training = time.time() - time_start
            output_result(model, X_train, y_train, X_test, y_test, time_training)
