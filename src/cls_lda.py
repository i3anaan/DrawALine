from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cls


def get_models(scenario):
    settings = []

    if args.small is not None:
        # Small
        setting = {}
        settings.append(setting)
    else:
        # Large
        setting = {}
        settings.append(setting)

    settings = cls.override_settings(args, settings, SVC)
    models = cls.models_from_settings(settings, SVC)
    return models


def declare_settings(subparser):
    return
