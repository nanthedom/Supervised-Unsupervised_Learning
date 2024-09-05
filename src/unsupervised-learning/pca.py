import numpy as np

class PCAScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, X):
        X = X - np.mean(X, axis=0)

        cov_matrix = np.cov(X, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components_ = eigenvectors[:, :self.n_components]

        total_variance = np.sum(eigenvalues)
        self.explained_variance_ = eigenvalues[:self.n_components] / total_variance

    def transform(self, X):
        X = X - np.mean(X, axis=0)
        return np.dot(X, self.components_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)