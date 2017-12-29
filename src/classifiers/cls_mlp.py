from sklearn.neural_network import MLPClassifier

defaults = {
    'solver': 'adam',
    'alpha': 0.03,
    'hidden_layer_sizes': (800, 10),
    'random_state': 1,
    'max_iter': 10000
}


def create(params=defaults):
    return MLPClassifier(params)
