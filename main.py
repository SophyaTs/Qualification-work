from clusterization import cluster, Method, ObjFunction, get_random, kmeans_pp_centroids
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

# assume that one row is coordinates of one center


def closest_centroid_index(m, x):
    d = [np.linalg.norm(m-xi) for xi in x]
    return np.argmin(d)


ITEM_COLORS = {
    0: "#FC3131",
    7: "#FD9433",
    1: "#FBDC2E",
    6: "#8EFF15",
    4: "#229430",
    5: "#34E0FD",
    3: "#3556FF",
    2: "#933FFA",
    8: "#FF46DD",
    9: "#808080",
}

CENTER_COLORS = {
    0: "#9F0808",
    7: "#A75101",
    1: "#D9A700",
    6: "#64BF01",
    4: "#035F0E",
    5: "#01A6C3",
    3: "#03177A",
    2: "#44078E",
    8: "#CE02A1",
    9: "#0B0B0B",
}


def draw(M, c, plot_centers=True):
    def item_color(m, x):
          i = closest_centroid_index(m, x)
          return ITEM_COLORS[i]

    def center_color(i):
          return CENTER_COLORS[i]
    n = M.shape[0]
    L = M.copy() if not plot_centers else np.concatenate((M, c), axis=0)

    if L.shape[1] > 2:
        L = (L - L.min())/(L.max()-L.min())
        pca = sklearnPCA(n_components=2)
        L = pca.fit_transform(L)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(L[0:n, 0], L[0:n, 1], c = [item_color(m, c) for m in M])
    if plot_centers:
        k = c.shape[0]
        plt.scatter(L[n:, 0], L[n:,1], c = [center_color(i) for i in range(k)])
    plt.show()


iris = load_iris().data
M = iris

# number of clusters
k = 3
#e = 0.1

M_sample = M

x_start = get_random(M, k)
x_start = kmeans_pp_centroids(M, k)

draw(M_sample, x_start, True)

x_min = cluster(
    M,
    x_start,
    Method.ralg,
    ObjFunction.eu
)

draw(M_sample, x_min, True)
