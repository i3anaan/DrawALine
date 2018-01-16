import feat_pca
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def qdc_pca(X_train, y_train, X_test, y_test, output_result):
    for k_PCA in range(40, 61, 1):
        X_train_feat, X_test_feat = feat_pca.pca(X_train, X_test, k_PCA)
        testAccuracy(X_train_feat, y_train, X_test_feat, y_test, output_result)

def testAccuracy(X_train, y_train, X_test, y_test, output_result):
    model = QuadraticDiscriminantAnalysis()
    model.fit(X_train, y_train)
    output_result(model, X_train, y_train, X_test, y_test)