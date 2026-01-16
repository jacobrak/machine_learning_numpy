import pandas as pd
import numpy as np
import math
class LinearRegression:
    def __init__(self):
        self.X = None
        self.Y = None
        self.beta = []

    def fit(self, X, y, intercept=True):
        if intercept:
            ones = np.ones((X.shape[0], 1))

        X = np.c_[ones, X]
        y = np.array(y)
        
        
        self.beta = np.linalg.inv(X.T @ X) @ X.T @y
        self.X = X
        self.Y = y

    def predict(self, x_test):
        y = x_test @ self.beta[1:] + self.beta[0]
        return y 
    
    def fit_lasso(self, _lambda=0.1, lr=0.01, epochs=1000):
        """ This is the iterative version to find L1 betas """
        n_samples, n_features = self.X.shape
        # Initialize beta with zeros if not already set
        for i in range(epochs):
            # 1. Prediction
            y_pred = self.X @ self.beta
            
            # 2. Error
            error = y_pred - self.Y
            
            # 3. Gradient Calculation
            # We treat the intercept (beta[0]) and slopes (beta[1:]) differently
            grad = (1 / n_samples) * (self.X.T @ error)
            
            # 4. Add L1 Penalty to the slopes ONLY
            # We don't penalize index 0
            penalty = _lambda * np.sign(self.beta)
            penalty[0] = 0 
            
            # 5. Update
            self.beta = self.beta - lr * (grad + penalty)
            if i % 500 == 0:
                print(self.beta)

    def get_l1_cost(self, _lambda):
        """ Your L1 cost logic moved to a helper function """
        n = self.X.shape[0]
        error = self.Y - (self.X @ self.beta)
        rss = (1 / (2 * n)) * np.linalg.norm(error)**2
        l1_penalty = _lambda * np.linalg.norm(self.beta[1:], ord=1)
        return rss + l1_penalty

if "__main__" == __name__:
    df = pd.read_csv("df.csv", index_col=0)
    X = df[["X1", "X2", "X3"]]
    y = df[["y"]]
    reg = LinearRegression()
    reg.fit(X, y)
    print(reg.predict([2, 3, 2]))
    reg.fit_lasso()
    print(reg.predict([2, 3, 2]))
    