from sklearn.neural_network import MLPClassifier
import cls


def get_models(args):
    settings = []

    if args.small is not None:
        # Small
        setting = {
            'alpha': 3**-4,
            'hidden_layer_sizes': (800, 200, 30),
            'random_state': 1,
            'activation': 'tanh',
            'max_iter': 10000
        }
        settings.append(setting)
    else:
        setting = {
            'alpha': 3**-4,
            'hidden_layer_sizes': (800, 200, 30),
            'random_state': 1,
            'activation': 'tanh',
            'max_iter': 10000
        }
        settings.append(setting)

    settings = cls.override_settings(args, settings, MLPClassifier)
    models = cls.models_from_settings(settings, MLPClassifier)
    return models


def declare_settings(subparser):
    subparser.add_argument('--alpha', help='The alpha setting', action='store', type=float)
    subparser.add_argument('--activation', help='The activation setting', action='store', choices=['identity', 'logistic', 'tanh', 'relu'])
    # TODO: How to add hidden layer sizes here?
    return
