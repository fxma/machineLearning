from matplotlib.pylab import array,diag
import matplotlib.pyplot as plt
import matplotlib as mpl
import pypr.clustering.gmm as gmm
import pypr.clustering.kmeans as kmeans
import numpy as np
from scipy import linalg

from sklearn import mixture

# 将样本点显示在二维坐标中
def plot_results(X, Y_, means, covariances, colors, eclipsed, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, colors)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        if eclipsed:
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

#创建样本数据，一共3个簇
mc = [0.4, 0.4, 0.2]
centroids = [ array([0,0]), array([3,3]), array([0,4]) ]
ccov = [ array([[1,0.4],[0.4,1]]), diag((1,2)), diag((0.4,0.1)) ]

X = gmm.sample_gaussian_mixture(centroids, ccov, mc, samples=1000)

#用plot_results函数显示未经聚类的样本数据
gmm = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, ['grey'],False, 0, "Sample data")

#用EM算法的GMM将样本分为3个簇，并按不同颜色显示在二维坐标系中
gmm = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, ['red', 'blue', 'green'], True, 1, "GMM clustered data")

plt.show()

