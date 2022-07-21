import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def show_img(img):
    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

def load_img(path):
    with Image.open(path) as im:
        img = np.asarray(im)
    return img

def save_img(img, path):
    Image.fromarray(img.astype(np.uint8)).save(path)

