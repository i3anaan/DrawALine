import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_models(scenario):
    if scenario=='small':
        settings = {
        }
    else:
        settings = {
        }
    return [LinearDiscriminantAnalysis(**settings)]
