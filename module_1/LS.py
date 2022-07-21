import numpy as np

def make_X(vec_src, vec_dst):
    _, n = vec_src.shape
    X = np.zeros((n * 2, 8))
    X[0::2, 0:3] = vec_src.T
    X[1::2, 3:6] = vec_src.T
    x_src, y_src, _ = vec_src
    x_dst, y_dst, _ = vec_dst
    X[0::2, 6] = - x_src * x_dst
    X[1::2, 6] = - x_src * y_dst
    X[0::2, 7] = - y_src * x_dst
    X[1::2, 7] = - y_src * y_dst
    return X

def make_y(vec_dst):
    return vec_dst[:2].T.reshape((-1, 1))

def solve_LS(X, y):
    Xt = X.T
    XtX = np.matmul(Xt, X)
    XtX_i = np.linalg.inv(XtX)
    XtX_i_Xt = np.matmul(XtX_i, Xt)
    return np.matmul(XtX_i_Xt, y)

def make_H(vec_H):
    return np.concatenate((vec_H[:, 0], [1]), axis=0).reshape(3, 3)

def find_H(vec_src, vec_dst):
    X = make_X(vec_src, vec_dst)
    y = make_y(vec_dst)
    vec_H = solve_LS(X, y)
    H = make_H(vec_H)
    return H

def transform_P(P, H):
    Pt = np.matmul(H, P)
    return Pt[:] / Pt[2]


