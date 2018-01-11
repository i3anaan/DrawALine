from sklearn.svm import SVC


def testAccuracy(X_train, y_train, X_test, y_test):
    print("Training SVC...")
    model = SVC(C=3)  # 10
    #model_reg = (lambda r: SVC(C=r))  # 10

    model.fit(X_train, y_train)
    print("Accuracy of the model on training: " +
          str(model.score(X_train, y_train)) + " and test: " +
          str(model.score(X_test, y_test)) + " data.")
