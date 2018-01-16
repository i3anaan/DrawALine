import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

def edit_distance_dissimilarity_single(x1, x2):
    xor = np.logical_xor(x1,x2)
    return (xor[xor == True].shape[0])/(xor.shape[0])

def edit_distance_dissimilarity(X_train, X_test):
    X_train_new = np.zeros((X_train.shape[0], X_train.shape[0]))
    X_test_new = np.zeros((X_test.shape[0], X_train.shape[0]))
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[0]):
            X_train_new[j,i] = edit_distance_dissimilarity_single(X_train[i], X_train[j])
        for j in range(X_test.shape[0]):
            X_test_new[j,i] = edit_distance_dissimilarity_single(X_train[i], X_test[j])
    print("test: ", edit_distance_dissimilarity_single(X_train[0], X_train[0]))
    print("xtrainnew: ", X_train_new[0])
    return X_train_new, X_test_new

def norm_similarity(X_train, X_test, norm_ord):
    X_train_new = np.zeros((X_train.shape[0], X_train.shape[0]))
    X_test_new = np.zeros((X_test.shape[0], X_train.shape[0]))
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[0]):
            X_train_new[j,i] = np.linalg.norm(X_train[i] - X_train[j], ord=norm_ord)
        for j in range(X_test.shape[0]):
            X_test_new[j,i] = np.linalg.norm(X_train[i] - X_test[j], ord=norm_ord)
    X_train_new_normed = X_train_new/X_train_new.max(axis=0)
    X_test_new_normed = X_test_new/X_test_new.max(axis=0)
    print("test: ", edit_distance_dissimilarity_single(X_train[0], X_train[0]))
    print("xtrainnew: ", X_train_new[0])
    return X_train_new_normed, X_test_new_normed

def cosine_similarity_all(X_train, X_test):
    X_train_new = cosine_similarity(X_train, X_train)
    X_test_new = cosine_similarity(X_test, X_train)
    return X_train_new, X_test_new
