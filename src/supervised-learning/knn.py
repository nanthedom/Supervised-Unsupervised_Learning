import numpy as np
from collections import Counter

class KNNScratch:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        
        elif self.metric == 'minkowski':
            """ - jika p = 1, maka Minkowski distance sama dengan Manhattan distance.
                - jika p = 2, maka Minkowski distance sama dengan Euclidean distance.
                - jika p > 3, jaraknya akan lebih sensitif terhadap perbedaan besar antara komponen x1 dan x2.
            """
            p = 3 
            return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)
        
        else:
            raise ValueError("Invalid distance metric")
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self._distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])

        return predictions
