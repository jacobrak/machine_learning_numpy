import numpy as np
import pandas as pd

# 1. Set a seed for reproducibility
np.random.seed(42)
def _create_dataset(    
    n_samples = 1000,
    beta0 = 5,      # Intercept
    beta1 = 2.5,    # Coefficient for X1
    beta2 = -1.2    # Coefficient for X2
    ):


    X = np.random.uniform(0, 10, size=(n_samples, 3))

    # y = beta0 + (beta1 * X1) + (beta2 * X2) + noise
    noise = np.random.normal(0, 1, n_samples)  # Adds some realism
    y = beta0 + (beta1 * X[:, 0]) + (beta2 * X[:, 1]) + noise

    df = pd.DataFrame({
        'X1': X[:, 0],
        'X2': X[:, 1],
        'X3': X[:, 2],
        'y': y
    })
    df.to_csv("df.csv")

if __name__ == "__main__":
    _create_dataset()