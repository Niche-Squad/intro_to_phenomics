import os
os.chdir("..")
import numpy as np
from matplotlib import pyplot as plt
from modules.image_handler import load_img

img = load_img("data/plots.jpg")
h, w, c = img.shape
img_vec = img.reshape((-1, 3))

mat_cor = np.corrcoef(img_vec)
i, j = 60, 130

idx = i * w + j
img_cor = mat_cor[idx, :].reshape((h, w))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[1].imshow(img_cor)
axes[1].set_title("Correlation Coef. Image")
fig.savefig("res/correlation.jpg", dpi=300)