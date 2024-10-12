# import rust_elastic_net
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from numba_implement import numba_elastic_net
from baseline import baseline_elastic_net


# Generate synthetic data
X, y = make_regression(n_samples=10000000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

baseline_coef, baseline_mse = baseline_elastic_net(X_train, y_train, X_test, y_test, alphas=[1.0], l1_ratio=0.5, max_iter=1000, tol=1e-4)
print("Coefficients from Scikit-learn Elastic Net:", baseline_coef)

# # Call the Rust implementation
# penalty = 0.3
# l1_ratio = 0.5
# coefficients = rust_elastic_net.elastic_net_cv(X_train.tolist(), y_train.tolist(), penalty, l1_ratio)

# print("Coefficients from Rust Elastic Net:", coefficients)

# Call the Numba implementation
numba_coef, numba_mse = numba_elastic_net(X_train, y_train, X_test, y_test, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4)
print(f'Numba Coefficients: {numba_coef}')

