import numpy as np

def kmeans(X, k, niter=20):
    n = len(X)
    centers = X[np.random.choice(n, k)]
    for i in range(niter):
        print("----- %d -----" % i)
        labels = assignment(X, centers)
        centers = update(X, labels, k)
    return dict(labels=labels, centers=centers)

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
