import os
os.chdir("module_2")
from gmm import gmm
from kmeans import kmeans
from image_handler import load_img, save_img
import matplotlib.pyplot as plt

img = load_img("bruges.jpeg")
plt.imshow(img)
h, w, c = img.shape

imgf = img.reshape((-1, 3))

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
k = 9
# Kmeans
out = KMeans(k).fit(imgf)
labels_k = out.labels_
centers = out.cluster_centers_
img_new_k = centers[labels_k].reshape((h, w, c))
save_img(img_new_k, "bruges_k%d.jpeg" % k)
# GMM
out = GaussianMixture(k).fit(imgf)
labels_g = out.predict(imgf)
centers = out.means_
img_new_g = centers[labels_g].reshape((h, w, c))
save_img(img_new_g, "bruges_g%d.jpeg" % k)

def plot_rgb(img, labels, filename, nr=3, nc=3):
    fig = plt.figure(figsize=(20, 20))
    for i in range(9):
        ax = fig.add_subplot(nr, nc, i + 1, projection='3d')
        ax.scatter(img[labels==i, 0],
                   img[labels==i, 1],
                   img[labels==i, 2],
                c=img[labels==i]/255, alpha=.7)
        ax.set_xlabel('red')
        ax.set_ylabel('green')
        ax.set_zlabel('blue')
        ax.set_xlim([0, 255])
        ax.set_ylim([0, 255])
        ax.set_zlim([0, 255])

    fig.tight_layout()
    fig.savefig(filename, transparent=False)

plot_rgb(imgf, labels_k, "rgb_k")
plot_rgb(imgf, labels_g, "rgb_g")


import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go
dt = pd.DataFrame(imgf)
dt.columns = ['x', 'y', 'z']
dt.loc[:, "color"] = ["rgb(%d, %d, %d)" % tuple(row.values.tolist())  for _, row in dt.iterrows()]

# trace = go.Scatter3d(
#         x=dt.x, y=dt.y, z=dt.z,
#         marker=dict(size=5,
#                     color=,
#                     ))
# data = [trace]
# layout = go.Layout(margin=dict(l=0,
#                                r=0,
#                                b=0,
#                                t=0))
# fig = go.Figure(data=data, layout=layout)
# fig.show()

fig = px.scatter_3d(dt, x='x', y='y', z='z',
                    color='color')
# fig.show()
plotly.offline.plot(fig, "fig.html")

fig