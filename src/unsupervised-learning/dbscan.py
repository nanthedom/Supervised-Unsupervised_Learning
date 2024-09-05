import numpy as np
from collections import deque

class DBSCAN:
    def __init__(self, epsilon=0.5, min_samples=5, metric='euclidean', p=2):
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.metric = metric
        self.p = p
        self.labels = None

    def calculate_distance(self, point1, point2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((point1 - point2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(point1 - point2))
        elif self.metric == 'minkowski':
            return np.sum(np.abs(point1 - point2) ** self.p) ** (1 / self.p)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def region_query(self, data, point_idx):
        neighbors = []
        for i in range(len(data)):
            if self.calculate_distance(data[point_idx], data[i]) <= self.epsilon:
                neighbors.append(i)
        return neighbors

    def expand_cluster(self, data, labels, point_idx, cluster_id):
        neighbors = self.region_query(data, point_idx)
        if len(neighbors) < self.min_samples:
            labels[point_idx] = -1  # label noise
            return False
        else:
            labels[point_idx] = cluster_id
            queue = deque(neighbors)
            while queue:
                neighbor_idx = queue.popleft()
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id  # ubah noise jadi bagian cluster
                elif labels[neighbor_idx] == 0:
                    labels[neighbor_idx] = cluster_id
                    new_neighbors = self.region_query(data, neighbor_idx)
                    if len(new_neighbors) >= self.min_samples:
                        queue.extend(new_neighbors)
            return True

    def fit(self, data):
        labels = [0] * len(data)  # label awal 0 = belum dikunjungi
        cluster_id = 0

        for point_idx in range(len(data)):
            if labels[point_idx] == 0:  # case titik belum dikunjungi
                if self.expand_cluster(data, labels, point_idx, cluster_id + 1):
                    cluster_id += 1

        self.labels = labels
        return labels