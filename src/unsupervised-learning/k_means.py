import numpy as np

class KMeansScratch:
    def __init__(self, n_clusters=3, max_iter=300, init_method='random'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init_method = init_method

    def fit(self, X):
        # Inisialisasi centroid
        self.centroids = self._initialize_centroids(X)
        self.labels = np.zeros(X.shape[0])
        
        for _ in range(self.max_iter):
            distances = self._compute_distances(X)
            self.labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def _initialize_centroids(self, X):
        if self.init_method == 'random':
            # Pilih centroid secara acak dari data
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            return X[indices]
        elif self.init_method == 'kmeans':
            # Inisialisasi dengan K-Means++
            centroids = np.empty((0, X.shape[1]))
            for _ in range(self.n_clusters):
                if centroids.shape[0] == 0:
                    centroid = X[np.random.choice(X.shape[0])]
                else:
                    distances = np.min(self._compute_distances(X, centroids), axis=1)
                    probs = distances**2 / np.sum(distances**2)
                    centroid = X[np.random.choice(X.shape[0], p=probs)]
                centroids = np.vstack([centroids, centroid])
            return centroids
        else:
            raise ValueError("Unknown initialization method: {}".format(self.init_method))

    def _compute_distances(self, X, centroids=None):
        if centroids is None:
            centroids = self.centroids
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def get_centroids(self):
        return self.centroids

    def get_labels(self):
        return self.labels
