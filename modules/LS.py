import numpy as np
import matplotlib.pyplot as plt

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

def plot_shape(P, label=""):
    """ visualize a closed shape defined by a list 2D points

    Parameters
    ----------
    P : array_like
        a NumPy array in a shape of (3, 4). The first axis should contain
        xy-coordinate values in an order of [x, y, 1]
    label : str, optional
        plot legend label, by default ""
    """
    x, y, _ = P
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    plt.plot(x, y, marker='o', label=label)
    plt.xlim(0, 50)
    plt.ylim(0, 50)

def move_shape(P, tx, ty):
    """Move points P by tx on x-axis and ty on y-axis

    Parameters
    ----------
    P : array_like
        a NumPy array in a shape of (3, 4). The first axis should contain
        xy-coordinate values in an order of [x, y, 1]
    tx : int
        move P by tx units on x-axis
    ty : int
        move P by ty units on y-axis

    Returns
    -------
    P_hat: array_like
        new coordinates that are moved from P by (tx, ty) units.
    """
    t = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]])
    return np.matmul(t, P)

def move_shape_v2(P, tx, ty, angle=0):
    radian = angle * np.pi / 180
    t = np.array([[np.cos(radian), -np.sin(radian), tx],
                  [np.sin(radian),  np.cos(radian), ty],
                  [             0,               0,  1]])
    return np.matmul(t, P)


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
