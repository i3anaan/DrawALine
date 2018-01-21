import json

def check_settings(args, classifier):
    override = {}
    for param in classifier().get_params():
        if param in vars(args):
            override[param] = vars(args)[param]
    print(override)
    return override

def override_settings(args, settings, classifier):
    override = check_settings(args, classifier)
    strings = []

    for setting in settings:
        setting = {**setting, **override}
        strings.append(json.dumps(setting, sort_keys=True))

    unique = list(set(strings))
    settings = []
    for setting in unique:
        settings.append(json.loads(setting))

    return settings


def models_from_settings(settings, classifier):
    models = []
    for setting in settings:
        models.append(classifier(**setting))

    return models
