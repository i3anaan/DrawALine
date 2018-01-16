from sklearn.decomposition import PCA

def pca(X_train, X_test, k):
    pca = PCA(n_components = k)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    return X_train, X_test
    #"""Reduce DATA using its K principal components."""
    #data = data.astype("float64")
    #data -= np.mean(data, axis=0)
    #U, S, V = np.linalg.svd(data, full_matrices=False)
    #return U[:, :k].dot(np.diag(S)[:k, :k])