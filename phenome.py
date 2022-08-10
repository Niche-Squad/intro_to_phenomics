import os
os.getcwd()
from modules.image_handler import load_img
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve2d

class Imager():

    def __init__(self, path):
        self.shape = None
        self.h = 0
        self.w = 0
        self.c = 0
        self.img = self.load(path)

    def load(self, path):
        return load_img(path)

    def show(self):
        plt.imshow(self.img)

    def get_blur(self):
        k_gauss = np.array((
            [1, 4, 1],
            [4, 9, 4],
            [1, 4, 1]),
            dtype='int') / 29
        return convolve2d(self.img[:, :, 0], k_gauss, mode="same")


obj = Imager("data/plots.jpg")
obj.img
obj_b = obj.get_blur()

plt.imshow(obj_b)



obj.load("path")

# public members
shape
h, w, c
img

# public functions
obj.get_shape() # returns a shape (dimension)
obj.blur() # returns a blurred image
obj.edge() # returns a edge-detected image
obj.show()



# private members
# private functions