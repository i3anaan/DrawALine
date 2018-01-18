import time
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def get_models(scenario):
    models = []
    if scenario=='small':
        settings = {
        }
        models.append(QuadraticDiscriminantAnalysis(**settings))
    else:
        settings = {
        }
        models.append(QuadraticDiscriminantAnalysis(**settings))

    return models
