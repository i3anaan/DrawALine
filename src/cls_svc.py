from sklearn.svm import SVC


def testAccuracy(X_train, y_train, X_test, y_test, output_result):
    print("Training SVC...")
    model = SVC(C=3)  # 10
    #model_reg = (lambda r: SVC(C=r))  # 10

    model.fit(X_train, y_train)

    output_result(model, X_train, y_train, X_test, y_test)
