import numpy as np
from collections import Counter

def calculate_entropy(labels):
    # hitung entropi dari array label.
    counts = np.bincount(labels)
    probabilities = counts / len(labels)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTreeScratch:
    def __init__(self, min_samples_split=2, max_depth=100, max_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        # Melatih model menggunakan data fitur X dan label y.
        num_features = X.shape[1]
        self.max_features = num_features if self.max_features is None else min(self.max_features, num_features)
        self.root = self._build_tree(X, y)

    def predict(self, X):
        # Memprediksi label untuk data fitur X.
        return np.array([self._predict_instance(x, self.root) for x in X])

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        features = np.random.choice(num_features, self.max_features, replace=False)
        best_feature, best_threshold = self._find_best_split(X, y, features)

        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        left_indices, right_indices = self._partition(X[:, best_feature], best_threshold)
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return TreeNode(best_feature, best_threshold, left_child, right_child)

    def _find_best_split(self, X, y, feature_indices):
        best_gain = -1
        best_feature, best_threshold = None, None
        for feature in feature_indices:
            column = X[:, feature]
            thresholds = np.unique(column)
            for threshold in thresholds:
                gain = self._calculate_information_gain(y, column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _calculate_information_gain(self, y, column, threshold):
        parent_entropy = calculate_entropy(y)
        left_indices, right_indices = self._partition(column, threshold)
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        
        num_samples = len(y)
        left_entropy = calculate_entropy(y[left_indices])
        right_entropy = calculate_entropy(y[right_indices])
        weighted_avg_child_entropy = (len(left_indices) / num_samples) * left_entropy + (len(right_indices) / num_samples) * right_entropy

        return parent_entropy - weighted_avg_child_entropy

    def _partition(self, column, threshold):
        left_indices = np.where(column <= threshold)[0]
        right_indices = np.where(column > threshold)[0]
        return left_indices, right_indices

    def _predict_instance(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_instance(x, node.left)
        else:
            return self._predict_instance(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
