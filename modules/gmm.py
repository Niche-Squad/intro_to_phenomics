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
    mu, sigma, pi = init_states(X, k)
    w = [1 / k] * k
    ll, ll_old = np.log(0), np.log(0)
    # EM iterations
    while ll - ll_old) > 3
        # E-step
        R = calculate_R(X, pi, mu, sigma)
        # M-step
        pi    = update_pi(R)
        mu    = update_mu(R, X)
        sigma = update_sigma(R, X, mu)
        # likelihood
        lod = get_ll(X, pi, mu, sigma)
        ll += [lod]
    # return
    return dict(labels=R.argmax(axis=1),
                mu=mu, sigma=sigma,
                ll=ll)

def pdf_mvn(x, mu, sigma):
    """ Compute the probability P(x | MVN(mu, sigma))

    Parameters
    ----------
    x : array_like
        the inspected data.
        A NumPy array in a shape of (p,), where p is the number of data dimension.
    mu : array_like
        the mean of the mvn distribution.
        A NumPy array in a shape of (p,), where p is the number of data dimension.
    sigma : array_like
        the standard deviation of the mvn distribution.
        A NumPy array in a shape of (p, p), where p is the number of data dimension.

    Returns
    -------
    a floating number
        P(x | MVN(mu, sigma))

    Examples
    ---------
    x  = [10, 20]
    mu = [10, 20]
    sigma = [[10, 5],
             [5, 25]]
    pdf_mvn(x, mu, sigma) # should return 0.010610
    """
    # numerator
    e1  = np.matmul((x - mu).T, np.linalg.inv(sigma))
    e2  = np.matmul(e1, (x - mu))
    num = np.exp(-0.5 * e2)
    # denominator
    den = (np.linalg.det(sigma) * (2 * np.pi) ** len(x)) ** (1 / 2)
    # probability
    p = num / den
    return p


def init_states(X, k):
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
def calculate_R(X, pi, mu, sigma):
    """Calculate posterior probability, P(xi|cj)

    Parameters
    ----------
    X : array_like
        A NumPy array with a shape of (n, p),
        where n is the number of observations,
        and p is the data dimension.
    pi : array_like
        A NumPy array with a shape of (k,).
        Prior, P(A).
    mu : array_like
        A Numpy array with a shape of (k, p)
        The mean of the inspected MVN distribution
    sigma : array_like
        A Numpy array with a shape of (k, p, p)
        The covariance of the inspected MVN distribution

    Returns
    -------
    array_like, a NumPy array with a shape of (n, k),
    where n is the number of observations,
    and k is the number of clusters.
    """
    # create an empty R matrix
    n, k = len(X), len(mu)
    R = np.zeros((n, k))
    # numerator
    for i in range(n):
        for j in range(k):
            R[i, j] = pdf_mvn(X[i], mu[j], sigma[j]) * pi[j]
    # denominator
    R /= np.sum(R, axis=1)[:, None]
    return R

def update_pi(R):
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


# plotly
# colors     = ["rgb(%d, %d, %d)" % tuple(color.tolist()) for color in centers]
# dfs        = df.iloc[::100]
# labels_sub = labels[::100]
# ls_points  = []
# for i in range(3):
#     ls_points += [go.Scatter3d(x=dfs.loc[labels_sub==i, "red"],
#                                y=dfs.loc[labels_sub==i, "green"],
#                                z=dfs.loc[labels_sub==i, "blue"],
#                                mode='markers', name='cluster_%d' % (i + 1),
#                                marker=dict(color=colors[i], size=5,
#                                            symbol='circle', opacity=0.7))]
# # layout
# layout = go.Layout(scene=dict(xaxis=dict(title="red"),
#                               yaxis=dict(title="green"),
#                               zaxis=dict(title="blue")),
#                    margin=dict(l=100, r=200, b=0, t=0))
# # show
# fig = go.Figure(data=ls_points, layout=layout)
# fig.show()

# references:
# https://www.cs.cmu.edu/~epxing/Class/10715/lectures/EM.pdf
# https://www.youtube.com/watch?v=qMTuMa86NzU
# https://en.wikipedia.org/wiki/Multivariate_normal_distribution