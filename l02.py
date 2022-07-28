import numpy as np
import matplotlib.pyplot as plt

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
