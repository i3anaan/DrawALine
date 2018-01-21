from sklearn.svm import SVC


def get_models(scenario):
    models = []
    if scenario == 'small':
        for gamma in range(-3, 2):
            for C in range(-5, 6):
                settings = {'kernel': 'rbf', 'gamma': 10**gamma, 'C': 3**C}
                models.append(SVC(**settings))
    else:
        # Large
        for gamma in range(-3, 2):
            for C in range(-5, 6):
                settings = {'kernel': 'rbf', 'gamma': 10**gamma, 'C': 3**C}
                models.append(SVC(**settings))

    return models
