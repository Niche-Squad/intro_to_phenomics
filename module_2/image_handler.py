import numpy as np
from PIL import Image

def load_img(path):
    with Image.open(path) as im:
        img = np.asarray(im)
    return img

def save_img(img, path):
    Image.fromarray(img.astype(np.uint8)).save(path)

