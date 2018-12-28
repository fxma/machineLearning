from sklearn.cluster import SpectralClustering
import numpy as np
def sc(X):
    sc1 = SpectralClustering(n_clusters=2, assign_labels="kmeans")
    sc1.fit(X)
    print('spectral clustering(kmeans): ', sc1.labels_)

    w, h = len(X), len(X)
    #构建相似度矩阵，任意两个样本间的相似度= 100 - 两个样本的欧氏距离
    Matrix = [[100 - np.linalg.norm(X[x] - X[y]) for x in range(w)] for y in range(h)]

    sc2 = SpectralClustering(n_components=2, affinity="precomputed", assign_labels="discretize")
    sc2.fit(Matrix)
    print('spectral clustering(discretize): ', sc2.labels_)
    return

X = np.array([[185.4, 72.6], [155.0, 54.4], [170.2, 99.9], [172.2, 97.3], [157.5, 59.0], [190.5, 81.6], [188.0, 77.1], [167.6, 97.3], [172.7, 93.3], [154.9, 59.0]])
sc(X)