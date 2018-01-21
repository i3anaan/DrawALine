from sklearn.linear_model import LogisticRegression


def get_models(scenario):
    models = []
    if scenario == 'small':
        for C in range(-5, 6):
            settings = {'max_iter': 1000, 'C': 3**C}
            models.append(LogisticRegression(**settings))
    else:
        # Large
        for C in range(-5, 6):
            settings = {'max_iter': 1000, 'C': 3**C}
            models.append(LogisticRegression(**settings))

    return models
