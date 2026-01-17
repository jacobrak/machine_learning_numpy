import pandas as pd
import numpy as np
from _learning_decay import _sqrt_decay
import math
class LinearRegression:
    def __init__(self):
        self.X = None
        self.Y = None
        self.beta = []
        self.intercept_bool = False

    def fit(self, X, y, intercept=True):
        if intercept:
            ones = np.ones((X.shape[0], 1))
            self.intercept_bool = True
        X1 = np.c_[ones, X]
        y = np.array(y)
        
        self.beta = np.linalg.inv(X1.T @ X1) @ X1.T @y
        self.X = X1
        self.Y = y

    def mean(self, y):
        mean_v = 0
        for y_value in y:
            mean_v += y_value
        return y_value/len(y)

    def predict(self, x_test):
        y_hat = x_test @ self.beta[1:] + self.beta[0]
        return y_hat 
    
    def fit_lasso(self, _lambda=0.1, lr=0.01, epochs=1000, lr_decay=_sqrt_decay, lambda_optim=False):
        """ 
        fit_lasso uses gradient descentto minimize the cost functions technically incorrect, better apporaches exist but easiest to implemnet.
        """
        global_loss = math.inf
        best_beta, start_beta = self.beta.copy(), self.beta.copy()
        n_samples, n_features = self.X.shape
        
        if lambda_optim:
            lambda_params = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
        else:
            lambda_params = [_lambda]
        for lambdas in lambda_params:
            self.beta = start_beta.copy()
            lowest_loss = self.get_l1_cost(lambdas)
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
                penalty = lambdas * np.sign(self.beta)
                if self.intercept_bool:
                    penalty[0] = 0 
                
                # 5. Update Learning rate
                eta = lr_decay(lr, i)
                
                # 6. Update beta
                self.beta = self.beta - eta * (grad + penalty)
                
                # 7. current loss
                curr_loss = self.get_l1_cost(lambdas)
                
                if lowest_loss > curr_loss:
                    lowest_loss = curr_loss
                    if global_loss > lowest_loss:
                        global_loss = lowest_loss
                        best_beta = self.beta.copy()
                        #print(f"{global_loss}here")
                if i % 500 == 0:
                    pass
                    #print(self.beta)
                    #print(eta)
        # Update best parameter
        self.beta = best_beta

    def get_l1_cost(self, _lambda):
        """ Your L1 cost logic moved to a helper function """
        n = self.X.shape[0]
        error = self.Y - (self.X @ self.beta)
        rss = (1 / (2 * n)) * np.linalg.norm(error)**2
        l1_penalty = _lambda * np.linalg.norm(self.beta[1:], ord=1)
        return rss + l1_penalty
    
    def fit_ridge(self, _lambda=0.1, lr=0.01, epochs=1000, lr_decay=_sqrt_decay, lambda_optim = False):
        """ This is the iterative version to find L2 betas """
        global_loss = math.inf
        best_beta, start_beta = self.beta.copy(), self.beta.copy()
        n_samples, n_features = self.X.shape
        
        if lambda_optim:
            lambda_params = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
        else:
            lambda_params = [_lambda]

        for lambdas in lambda_params:
            self.beta = start_beta.copy()
            lowest_loss = self.get_l2_cost(lambdas)
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
                penalty = lambdas * 2*self.beta
                if self.intercept_bool:
                    penalty[0] = 0 

                # 5. Update Learning rate
                eta = lr_decay(lr, i)
                
                # 5. Update beta
                self.beta = self.beta - eta * (grad + penalty)
                
                # 7. current loss
                curr_loss = self.get_l2_cost(lambdas)

                if lowest_loss > curr_loss:
                    #print(f"lambda:{lambdas}, Curr_loss:{curr_loss}")
                    lowest_loss = curr_loss
                    if global_loss > lowest_loss:
                        
                        global_loss = lowest_loss
                        best_beta = self.beta.copy()
                        #print(f"{global_loss}here")
                if i % 500 == 0:
                    pass
                    #print(best_beta)
                    #print(eta)
        # Update best parameter
        self.beta = best_beta

    def get_l2_cost(self, _lambda=0.1):
        n = self.X.shape[0]
        error = self.Y - (self.X @ self.beta)
        rss = (1 / (2 * n)) * np.linalg.norm(error)**2
        l2_penalty = _lambda * np.linalg.norm(self.beta[1:], ord=2)**2
        return rss + l2_penalty
    
    def estimatebeta(self):
        return self.beta
    
    def RSS(self, X, y):
        """Residual sum of squares"""
        y_hat = self.predict(X)
        return np.sum((y - y_hat)**2)

    def TSS(self, y):
        """Total sum of squares"""
        mu = self.mean(y)
        return np.sum((y - mu)**2)
    
    def score(self, X, y):
        try: 
            rss = self.RSS(X, y)
            tss = self.TSS(y)
            
            # R^2 Formula: 1 - (Residual Sum of Squares / Total Sum of Squares)
            r2 = 1 - (rss / tss)
            return r2
        except ZeroDivisionError:
            print("Warning: TSS is zero, R2 cannot be calculated.")
            return 0.0  # Or raise the error depending on your needs
    
    def summary(self, X, y):
        n = len(y)
        r2 = self.score(X, y)
        mse = self.RSS(X, y) * 1/n
        rmse = np.sqrt(mse)
        y_mu = float(self.mean(y))
        return (f"R^2:{r2:.2f} \nMSE:{mse:.2f} \nRMSE:{rmse:.2f} \nymean:{y_mu:.2f}")

if "__main__" == __name__:
    df = pd.read_csv("df.csv", index_col=0)
    X = np.array(df[["X1", "X2", "X3", "X4", "X5"]])
    y = np.array(df[["y"]])
    reg = LinearRegression()
    reg.fit(X, y)
    print(reg.score(X, y))
    #reg.fit_ridge(lr=0.01, _lambda = 0.1,epochs=1000, lambda_optim=True)
    reg.fit_lasso(epochs=1000, lr=0.01, _lambda=0.1)
    print(reg.summary(X, y))
    exit()
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=0.1) 
    model.fit(X, y)

    # 3. Print the results
    print(f"Intercept (Beta 0): {model.intercept_}")
    print(f"Coefficients (Beta 1-4): {model.coef_}")

    # 4. Dictionary view for clarity
    coef_dict = dict(zip(X.columns, model.coef_))
    print("\nBreakdown:")
    print(coef_dict)