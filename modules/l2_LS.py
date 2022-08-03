import numpy as np

def make_X(P_src, P_dst):
    _, n = P_src.shape
    X = np.zeros((n * 2, 8))
    X[0::2, 0:3] = P_src.T
    X[1::2, 3:6] = P_src.T
    x_src, y_src, _ = P_src
    x_dst, y_dst, _ = P_dst
    X[0::2, 6] = - x_src * x_dst
    X[1::2, 6] = - x_src * y_dst
    X[0::2, 7] = - y_src * x_dst
    X[1::2, 7] = - y_src * y_dst
    return X

def make_y(P_dst):
    return P_dst[:2].T.reshape((-1, 1))

def solve_OLS(X, y):
    """Find the beta vector in OLS problem given the matices X and y.

    Parameters
    ----------
    X : array_like
        a NumPy array in a shape of (2 * n, 8), where n is the number of coordinates.
    y : array_like
        a NumPy array in a shape of (8,)

    Returns
    -------
    beta: array_like
        a vector that can minimize residuals in the OLS equation.
        The vector shape should be (2 * n,).
    """
    Xt = X.T
    XtX = np.matmul(Xt, X)
    XtX_i = np.linalg.inv(XtX)
    XtX_i_Xt = np.matmul(XtX_i, Xt)
    beta = np.matmul(XtX_i_Xt, y)
    return beta

def make_H(beta):
    return np.concatenate((beta[:, 0], [1]), axis=0).reshape(3, 3)

def find_H(P_src, P_dst):
    X = make_X(P_src, P_dst)
    y = make_y(P_dst)
    beta = solve_OLS(X, y)
    H = make_H(beta)
    return H

def transform_P(P, H):
    Pt = np.matmul(H, P)
    return Pt[:] / Pt[2]


# # P_hat * P.T * inv(P * P.T) = H
# x = [10, 30, 30, 10]
# y = [10, 10, 20, 20]
# i = [1] * 4
# P = np.array([x, y, i])
# # define P_hat
# x = [15, 25, 25, 10]
# y = [10, 15, 30, 30]
# P_hat = np.array([x, y, i])

# a = np.matmul(P_hat, P.T)
# b = np.matmul(P, P.T)
# bi = np.linalg.inv(b)
# h = np.matmul(a, bi)
# np.matmul(h, P)
