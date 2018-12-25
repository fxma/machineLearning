from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[185.4, 72.6], [155.0, 54.4], [170.2, 99.9], [172.2, 97.3], [157.5, 59.0], [190.5, 81.6], [188.0, 77.1], [167.6, 97.3], [172.7, 93.3], [154.9, 59.0]])
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
y_kmeans = kmeans.predict(X)
centroids = kmeans.cluster_centers_

# if __name__ == '__main__':
plt.scatter(X[:, 0], X[:, 1], s=50);
plt.yticks(())
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5);
plt.show()

print(kmeans.predict([[170.0, 60], [155.0, 50]]))
print(y_kmeans)
