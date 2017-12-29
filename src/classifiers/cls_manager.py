from . import cls_logistic
from . import cls_mlp
from . import cls_svm


def create_default_logistic(training_set):
    return cls_logistic.create().fit(*training_set)


def create_custom_logistic(training_set, params):
    pars = {**cls_logistic.defaults, **params}
    return cls_logistic.create(pars).fit(*training_set)


def create_default_mlp(training_set):
    return cls_mlp.create().fit(*training_set)


def create_custom_mlp(training_set, params):
    pars = {**cls_mlp.defaults, **params}
    print(pars)
    print(cls_mlp.create(pars))
    print(training_set)
    print(*training_set)
    return cls_mlp.create(pars).fit(*training_set)


def create_default_svm(training_set):
    return cls_svm.create().fit(*training_set)


def create_custom_svm(training_set, params):
    pars = {**cls_svm.defaults, **params}
    return cls_svm.create(pars).fit(*training_set)
