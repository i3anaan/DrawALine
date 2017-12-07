from sklearn.neighbors import NearestNeighbors
import numpy as np


defaults = {
    'n_neighbors' = 2,
    'weights' = 'uniform'
}


def create(params = defaults):
    return NearestNeighbors(params)
