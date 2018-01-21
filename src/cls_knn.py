from sklearn.neighbors import KNeighborsClassifier
import cls

# best values:
# k_NN = 3; k_PCA = 40 and 41 -> 97.80%
# k_NN = 1; k_PCA = 43 and 44 -> 97.80%
# without PCA: K_NN = 1 -> 96.00% very slow!
def get_models(scenario):
    settings = []

    if args.small is not None:
        # Small
        for weight in ['uniform', 'distance']:
            for k in range(1, 6, 1):
                setting = {
                    'n_neighbors': k,
                    'weights': weight,
                }
                settings.append(setting))
    else:
        # Large
        for weight in ['uniform', 'distance']:
            for k in range(1, 21, 1):
                setting = {
                    'n_neighbors': k,
                    'weights': weight,
                }
                settings.append(setting))

    settings = cls.override_settings(args, settings, SVC)
    models = cls.models_from_settings(settings, SVC)
    return models


def declare_settings(subparser):
    # No settings
