from sklearn.linear_model import LogisticRegression

defaults = {'max_iter': 1000, 'C': 10}


def create(params=defaults):
    return LogisticRegression(**params)
