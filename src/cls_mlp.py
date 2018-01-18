import time
from sklearn.neural_network import MLPClassifier

def get_models(scenario):
    models = []
    if scenario=='small':
        settings = {
            'alpha': 3**-4,  # 0.01
            'hidden_layer_sizes': (800, 200, 30),  # 800, 200, 30 -> 97.6
            'random_state': 1,
            'activation': 'tanh',
            'max_iter': 10000
        }
        models.append(MLPClassifier(**settings))
    else:
        settings = {
            'alpha': 3**-4,  # 0.01
            'hidden_layer_sizes': (800, 200, 30),  # 800, 200, 30 -> 97.6
            'random_state': 1,
            'activation': 'tanh',
            'max_iter': 10000
        }
        models.append(MLPClassifier(**settings))

    return models
