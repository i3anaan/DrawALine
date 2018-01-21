from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_models(scenario):
    models = []
    if scenario == 'small':
        settings = {}
        models.append(LinearDiscriminantAnalysis(**settings))
    else:
        # Large
        settings = {}
        models.append(LinearDiscriminantAnalysis(**settings))

    return models
