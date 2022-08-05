import numpy as np

def KMeans_sklearn(X, k):
    from sklearn.cluster import KMeans
    out = KMeans(k).fit(X)
    return dict(labels=out.labels_,
                centers=out.cluster_centers_)

def KMeans(X, k, niter=20):
    centers = init_center(X, k)
    for _ in range(niter):
        labels = assignment(X, centers)
        centers = update(X, labels, k)
    return dict(labels=labels, centers=centers)

def init_center(X, k):
    n = len(X)
    centers = X[np.random.choice(n, k)]
    return centers

def euclidean_distance(x, y):
    return np.sum((x - y) ** 2) ** .5

def assignment(X, centers):
    n = len(X)
    k = len(centers)
    distances = np.zeros((n, k))
    for i, x in enumerate(X):
        for j, center in enumerate(centers):
            distances[i, j] = euclidean_distance(x, center)
    labels = np.argmin(distances, axis=1)
    return labels

def update(X, labels, k):
    n, p = X.shape
    new_centers = np.zeros((k, p))
    for i in range(k):
        new_centers[i] = X[labels == i].mean(axis=0)
    return new_centers
