from sklearn.svm import SVC

defaults = {'C': 3}


def create(params=defaults):
    return SVC(params)
