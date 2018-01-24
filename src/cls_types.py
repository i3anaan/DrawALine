def get_type(cls):
    return {
        'lda': 'LinearDiscriminantAnalysis',
        'qda': 'QuadraticDiscriminantAnalysis',
        'knn': 'KNeighborsClassifier',
        'knn-pca': 'KNeighborsClassifier',
        'log': 'LogisticRegression',
        'mlp': 'MLPClassifier',
        'svc': 'SVC'
    }[cls] 

