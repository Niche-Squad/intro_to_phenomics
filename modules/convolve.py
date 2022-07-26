from image_handler import load_img, save_img, show_img
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

# start from a simple image
img = np.zeros((10, 10))
img[2:5, 2:5] = 1

kernel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]) / 8

imgc = np.zeros(img.shape)
for i in range(1, 9):
    for j in range(1, 9):
        mat = img[(i - 1):(i + 2), (j - 1):(j + 2)] * kernel_x
        imgc[i, j] = np.sum(mat)

imgc_x = convolve2d(img, kernel_x, mode="same")

fig = plt.figure(figsize=(10, 10))
for i, im in enumerate([img, abs(imgc), abs(imgc_x)]):
    ax = fig.add_subplot(3, 1, i + 1)
    ax.imshow(im, cmap="gray")
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])

kernel_y = np.array([[-1, -2, -1],
                     [0,   0,  0],
                     [1,   2,  1]]) / 8


imgc_y = convolve2d(img, kernel_y, mode="same")

img_new = abs(imgc_x) + abs(imgc_y)

fig = plt.figure(figsize=(10, 10))
for i, im in enumerate([img, img_new >= .5]):
    ax = fig.add_subplot(3, 1, i + 1)
    ax.imshow(im, cmap="gray")

def convolve_edge(img):
    k_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]) / 8
    k_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]) / 8
    imgc_x = convolve2d(img, k_x, mode="same")
    imgc_y = convolve2d(img, k_y, mode="same")
    img_new = ((imgc_x)**2 + (imgc_y)**2 )**.5
    return img_new

def convolve_edge(img):
    k = np.array([[-1 - 1j, 0 - 2j, 1 - 1j],
                  [-2 + 0j, 0 + 0j, 2 + 0j],
                  [-1 + 1j, 0 + 2j, 1 + 1j]]) / 8
    imgc = convolve2d(img, k, mode="same")
    return imgc

# img = load_img("../module_2/bruges.jpeg").mean(axis=2).astype(int)
img = load_img("sd.jpeg").mean(axis=2).astype(int)

imgc = convolve_edge(img)
fig = plt.figure(figsize=(20, 20))
# for i, im in enumerate([imgbw, imgc > np.quantile(imgc, .99)]):
for i, im in enumerate([img, abs(imgc)]):
    ax = fig.add_subplot(2, 1, i + 1)
    ax.imshow(im, cmap="gray")


k_edge = np.array((
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]),
    dtype='int')

imgc = convolve2d(img, k_edge, mode="same")
plt.imshow(abs(imgc), cmap="gray")

k_shp = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]),
    dtype='int')

imgc = convolve2d(img, k_shp, mode="same")
# imgc[imgc < 0] = 0
plt.imshow((imgc)[300:500, 400:600]**2, cmap="gray")
plt.imshow(img[300:500, 400:600], cmap="gray")

k_corner = np.array((
    [1, -2, 1],
    [-2,  4, -2],
    [1, -2, 1]),
    dtype='int')

imgc = convolve2d(img, k_corner, mode="same")
plt.imshow(abs(imgc), cmap="gray")

k_gauss = np.array((
    [1, 4, 1],
    [4, 9, 4],
    [1, 4, 1]),
    dtype='int') / 29

imgc = convolve2d(img, k_gauss, mode="same")
plt.imshow(imgc[300:500, 400:600], cmap="gray")
plt.imshow(img[300:500, 400:600], cmap="gray")


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

# https://setosa.io/ev/image-kernels/
