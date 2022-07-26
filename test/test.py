import numpy as np
import matplotlib.pyplot as plt

# Math(r'F(k) = \int_{-\infty}^{\infty} f(x) e^{2\pi i k} dx')

vec = np.array([[20, 20, 40, 40],
                [20, 40, 40, 20],
                [1] * 4])

x, y, i = vec
plt.plot(x, y)

plt.plot(x, y)
plt.xlim(0, 100)
plt.ylim(0, 100)

plt.plot(x, y, marker='o')
plt.xlim(0, 100)
plt.ylim(0, 100)

x = np.append(x, x[0])
y = np.append(y, y[0])
plt.plot(x, y, marker='o')
plt.xlim(0, 100)
plt.ylim(0, 100)


def plot_vec(vec, xlim=100, ylim=100):
    """_summary_

    Args:
        vec (_type_): _description_
        limit (int, optional): _description_. Defaults to 100.
    """
    x, y, i = vec
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    plt.plot(x, y, marker='o')
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)

tx = 30
ty = 20
# 3 by 3
H = [[1, 0, tx],
     [0, 1, ty],
     [0, 0, 1]]
# 4 by 3
mat_rect_new = np.matmul(H, vec)

plot_vec(vec)
plot_vec(mat_rect_new)


def move_xy(vec, x, y):
    """_summary_

    Args:
        vec (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    H = [[1, 0, x],
         [0, 1, y],
         [0, 0, 1]]
    vec_p = np.matmul(H, vec)
    return vec_p

plot_vec(vec)
vec2 = move_xy(vec, x=30, y=50)
plot_vec(vec2)


# holographic

v1 = np.array([[10, 15, 45, 40],
               [20, 40, 35, 25],
               [1] * 4])

v2 = np.array([[45, 50, 80, 95],
               [25, 60, 80, 60],
              [1] * 4])


plot_vec(v1)
plot_vec(v2)


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




import cv2 as cv


v1 = np.array([[240, 500, 3500, 3500],
               [0, 3000, 2450, 250],
               [1] * 4])
plt.imshow(img)
plot_vec(v1, xlim=img.shape[1], ylim=img.shape[0])



w = 5000
h = 3000
v2 = np.array([[0, 0, w, w],
               [0, h, h, 0],
               [1] * 4])

H = find_H(v1, v2)

img2 = cv.warpPerspective(img.astype(np.float32), H, (w, h))

Image.fromarray(img2.astype(np.uint8)).save("metro2.jpeg")


# REFERENCES
# https://towardsdatascience.com/understanding-homography-a-k-a-perspective-transformation-cacaed5ca17
# https://wordsandbuttons.online/interactive_guide_to_homogeneous_coordinates.html
# page 171: Inverse warping algorithm.
    """_summary_
    """