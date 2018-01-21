from sklearn.neural_network import MLPClassifier
import cls


def get_models(scenario):
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

    settings = cls.override_settings(args, settings, SVC)
    models = cls.models_from_settings(settings, SVC)
    return models


def declare_settings(subparser):
    # No settings
