import numpy as np

class GaussianNaiveBayesScratch:
    def __init__(self):
        self._mean = None
        self._var = None
        self._priors = None
        self._classes = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Inisialisasi mean, variansi, dan prior probability
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # Hitung mean, variansi, dan prior probability untuk setiap kelas
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / n_samples

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posteriors = []

        # Hitung posterior probability untuk setiap kelas
        for idx, c in enumerate(self._classes):
            prior_log = np.log(self._priors[idx])  # Log prior
            likelihood = np.sum(np.log(self._gaussian_density(idx, x)))  # Log likelihood
            posterior = prior_log + likelihood
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)] # posterior tertinggi

    def _gaussian_density(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        # Formula distribusi Gaussian untuk menghitung likelihood
        numerator = np.exp(-0.5 * ((x - mean) ** 2) / var)
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
