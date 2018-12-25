from sklearn.neighbors import KNeighborsClassifier

X = [[185.4, 72.6],
     [155.0, 54.4],
     [170.2, 99.9],
     [172.2, 97.3],
     [157.5, 59.0],
     [190.5, 81.6],
     [188.0, 77.1],
     [167.6, 97.3],
     [172.7, 93.3],
     [154.9, 59.0]]
y = [0, 1, 2, 2, 1, 0, 0, 2, 2, 1]

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

print(neigh.predict([[170.0, 60], [155.0, 50]]))