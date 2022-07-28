import numpy as np

def gmm_sklearn(X, k):
    from sklearn.mixture import GaussianMixture
    out = GaussianMixture(k).fit(X)
    return dict(labels=out.predict(X),
                centers=out.means_)

def gmm(X, k, niter=20):
    """_summary_

    Parameters
    ----------
    X : _type_
        _description_
    k : _type_
        _description_
    niter : int, optional
        _description_, by default 100

    Returns
    -------
    _type_
        _description_

    Examples
    --------
    >>> k = 3
    >>> n = 100
    >>> N = n * k
    >>> w = [1 / k] * k
    >>> ll = []
    >>> dist1 = multivariate_normal([10, 20], [[10, 5], [5, 25]]).rvs(size=n)
    >>> dist2 = multivariate_normal([50, 10], [[3, 1], [1, 5]]).rvs(size=n)
    >>> dist3 = multivariate_normal([5, 60], [[10, 0], [0, 15]]).rvs(size=n)
    >>> X = np.concatenate([dist1, dist2, dist3])
    """
    mu, sigma = init_random_dist(X, k)
    w = [1 / k] * k
    R = np.zeros((len(X), k))
    ll = []
    # EM iterations
    for i in range(niter):
        # E-step
        R = estimate_R(R, X, w, mu, sigma)
        # M-step
        w     = update_w(R)
        mu    = update_mu(R, X)
        sigma = update_sigma(R, X, mu)
        # likelihood
        lod = get_ll(X, w, mu, sigma)
        ll += [lod]
    # return
    return dict(labels=R.argmax(axis=1),
                mu=mu, sigma=sigma,
                ll=ll)

def pdf_mvn(x, mu, sigma):
    """_summary_

    Parameters
    ----------
        x (_type_): _description_
        mu (_type_): _description_
        sigma (_type_): _description_

    Returns
    -------
        _type_: _description_

    Examples
    ---------
    >>> from scipy.stats import multivariate_normal
    >>> x = [9, 10]
    >>> mu = [10, 20]
    >>> sigma = [[10, 5],
                 [5, 25]]
    >>> multivariate_normal.pdf(x, mean=mu, cov=sigma)
    >>> pdf_mvn(x, mu, sigma)
    """
    # numerator
    e1 = np.matmul((x - mu).T, np.linalg.inv(sigma))
    e2 = np.matmul(e1, (x - mu))
    num = np.exp(-0.5 * e2)
    # denominator
    den = (np.linalg.det(sigma) * (2 * np.pi) ** len(x)) ** (1 / 2)
    # probability
    p = num / den
    return p

def init_random_dist(X, k):
    """_summary_

    Parameters
    ----------
    X : _type_
        _description_
    k : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    n = len(X)
    # randomly assign a data point to a cluster
    clusters = np.random.random(n) // (1 / k)
    Xc    = [X[clusters == i] for i in range(k)]
    mu    = [np.mean(xc, axis=0) for xc in Xc]
    sigma = [np.cov(xc.T) for xc in Xc]
    return mu, sigma

# update
def estimate_R(R, X, w, mu, sigma):
    """_summary_

    Parameters
    ----------
    R : _type_
        _description_
    X : _type_
        _description_
    mu : _type_
        _description_
    sigma : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    n, k = R.shape
    for i in range(n):
        for j in range(k):
            R[i, j] = w[j] * pdf_mvn(X[i], mu[j], sigma[j])
    # to make it summed to one
    R /= np.sum(R, axis=1)[:, None]
    return R

def update_w(R):
    return R.mean(axis=0)

def update_mu(R, X):
    n, k = R.shape
    new_mu = [0] * k
    for j in range(k):
        num = np.sum(X * R[:, j][:, None], axis=0)
        den = np.sum(R[:, j])
        new_mu[j] = num / den
    return new_mu

def update_sigma(R, X, mu):
    n, k = R.shape
    new_sigma = [0] * k
    for j in range(k):
        # weighted mean squared devication
        weighted_MSD = [np.matmul((x - mu[j])[:, None], (x - mu[j])[:, None].T) * r for x, r in zip(X, R[:, j])]
        new_sigma[j] = np.sum(weighted_MSD, axis=0) / np.sum(R[:, j])
    return new_sigma

def get_ll(X, w, mu, sigma):
    n = len(X)
    k = len(w)
    mat_ll = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            mat_ll[i, j] = w[j] * pdf_mvn(X[i], mu[j], sigma[j])
    ll = np.log(np.sum(mat_ll, axis=1)).sum()
    return ll

# references:
# https://www.cs.cmu.edu/~epxing/Class/10715/lectures/EM.pdf
# https://www.youtube.com/watch?v=qMTuMa86NzU
# https://en.wikipedia.org/wiki/Multivariate_normal_distribution