from sklearn.linear_model import LogisticRegression


def testAccuracy(X_train, y_train, X_test, y_test, output_result):
    print("Running LOG...")

    for C in range(-5, 5):
        model = LogisticRegression(max_iter=1000, C=3**C)
        model.fit(X_train, y_train)
        output_result(model, X_train, y_train, X_test, y_test)
