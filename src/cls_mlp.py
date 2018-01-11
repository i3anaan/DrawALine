from sklearn.neural_network import MLPClassifier


def testAccuracy(X_train, y_train, X_test, y_test, output_result):
    print("Training MLP...")
    model = MLPClassifier(
        solver='adam',
        alpha=0.01,  # 0.01
        hidden_layer_sizes=(800, 200, 30),  # 800, 200, 30 -> 97.6
        random_state=1,
        max_iter=10000)

    model.fit(X_train, y_train)
    output_result(model, X_train, y_train, X_test, y_test)
