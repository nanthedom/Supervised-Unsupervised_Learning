import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, C=1.0, loss_function='cross_entropy'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.C = C
        self.loss_function = loss_function
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        
        # Gradient Descent
        for i in range(self.n_iterations):
            z = np.dot(X, self.W) + self.b
            y_pred = self.sigmoid(z)
            
            # itung gradients
            dw = (1 / self.m) * np.dot(X.T, (y_pred - y))
            db = (1 / self.m) * np.sum(y_pred - y)
            
            if self.regularization == 'l2':
                dw += (self.C / self.m) * self.W
            elif self.regularization == 'l1':
                dw += (self.C / self.m) * np.sign(self.W)
            
            # Update parameters
            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            if i % 100 == 0:
                cost = self.compute_cost(X, y)
                print(f"Iteration {i}, Cost: {cost}")

    def predict(self, X):
        y_pred = self.sigmoid(np.dot(X, self.W) + self.b)
        return np.round(y_pred)
    
    def compute_cost(self, X, y):
        m = len(y)
        h = self.sigmoid(np.dot(X, self.W) + self.b)
        if self.loss_function == 'cross_entropy':
            cost = -1/m * np.sum(y * np.log(h + 1e-8) + (1 - y) * np.log(1 - h + 1e-8))
        elif self.loss_function == 'mse':
            cost = 1/(2*m) * np.sum((h - y) ** 2)
        
        if self.regularization == 'l2':
            cost += self.C / (2 * m) * np.sum(self.W ** 2)
        elif self.regularization == 'l1':
            cost += self.C / m * np.sum(np.abs(self.W))
        
        return cost
