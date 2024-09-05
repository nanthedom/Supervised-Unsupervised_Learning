import numpy as np
from cvxopt import matrix, solvers

class LinearSVC:
    def __init__(self, optimizer='quadratic', kernel='linear', C=1.0):
        self.optimizer = optimizer
        self.kernel = kernel
        self.C = C
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.w = None
        self.b = None

    def fit(self, X, y):
        if self.optimizer == 'quadratic' and self.kernel == 'linear':
            m, n = X.shape

            # linear kernel
            K = np.dot(X, X.T)

            # quadratic
            P = matrix(np.outer(y, y) * K)
            q = matrix(-np.ones(m))
            G = matrix(np.vstack([-np.eye(m), np.eye(m)]))
            h = matrix(np.hstack([np.zeros(m), np.ones(m) * self.C]))
            A = matrix(y, (1, m), 'd')
            b = matrix(0.0)

            sol = solvers.qp(P, q, G, h, A, b)
            self.alpha = np.array(sol['x']).flatten()

            # hitung weights & bias
            support_vector_indices = self.alpha > 1e-5
            self.support_vectors = X[support_vector_indices]
            self.support_vector_labels = y[support_vector_indices]
            self.alpha = self.alpha[support_vector_indices]

            self.w = np.dot(self.alpha * self.support_vector_labels, self.support_vectors)
            self.b = np.mean(self.support_vector_labels - np.dot(self.support_vectors, self.w))
        
        else:
            print('Error: kernel and optimizer not supported!')

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)