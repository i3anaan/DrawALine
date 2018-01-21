from sklearn.svm import SVC
import cls


def get_models(args):
    settings = []

    if args.small is not None:
        # Small
        for gamma in range(-3, 2):
            for C in range(-5, 6):
                setting = {'kernel': 'rbf', 'gamma': 10**gamma, 'C': 3**C}
                settings.append(setting)
    else:
        # Large
        for gamma in range(-3, 2):
            for C in range(-5, 6):
                setting = {'kernel': 'rbf', 'gamma': 10**gamma, 'C': 3**C}
                settings.append(setting)

    settings = cls.override_settings(args, settings, SVC)
    models = cls.models_from_settings(settings, SVC)
    return models

def declare_settings(subparser):
    subparser.add_argument('--gamma', help='The gamma setting', action='store', type=float)
    subparser.add_argument('--C', help='The C setting', action='store', type=float)
    subparser.add_argument('--kernel', help='The kernel setting', action='store', choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
    return
