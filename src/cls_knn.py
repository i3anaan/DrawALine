import sys
import numpy as np
import math
import time
import feat_pca

from sklearn.neighbors import KNeighborsClassifier


# best values:
# k_NN = 3; k_PCA = 40 and 41 -> 97.80%
# k_NN = 1; k_PCA = 43 and 44 -> 97.80%
# without PCA: K_NN = 1 -> 96.00% very slow!
def get_models(scenario):
    models = []
    if scenario == 'small':
        for weight in ['uniform', 'distance']:
            for k in range(1, 6, 1):
                settings = {
                    'n_neighbors': k,
                    'weights': weight,
                }
                models.append(KNeighborsClassifier(**settings))
    else:
        # Large
        for weight in ['uniform', 'distance']:
            for k in range(1, 21, 1):
                settings = {
                    'n_neighbors': k,
                    'weights': weight,
                }
                models.append(KNeighborsClassifier(**settings))

    return models
