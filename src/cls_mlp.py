from sklearn.neural_network import MLPClassifier


def testAccuracy(X_train, y_train, X_test, y_test):
    print("Training MLP...")
    model = MLPClassifier(
        solver='adam',
        alpha=0.01,  # 0.01
        hidden_layer_sizes=(800, 200, 30),  # 800, 200, 30 -> 97.6
        random_state=1,
        max_iter=10000)

    model.fit(X_train, y_train)
    print("Accuracy of the model on training: " +
          str(model.score(X_train, y_train)) + " and test: " +
          str(model.score(X_test, y_test)) + " data.")
