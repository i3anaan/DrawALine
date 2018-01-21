from sklearn.linear_model import LogisticRegression
import cls


def get_models(scenario):
    settings = []

    if args.small is not None:
        # Small
        for C in range(-5, 6):
            setting = {'max_iter': 1000, 'C': 3**C}
            settings.append(setting)
    else:
        # Large
        for C in range(-5, 6):
            setting = {'max_iter': 1000, 'C': 3**C}
            settings.append(setting)

    settings = cls.override_settings(args, settings, SVC)
    models = cls.models_from_settings(settings, SVC)
    return models


def declare_settings(subparser):
    subparser.add_argument('--C', help='The C setting', action='store', type=float)
    return
