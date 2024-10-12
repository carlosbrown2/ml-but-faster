import numpy as np
from numba import jit
from common import timed_evaluation


@jit(cache=True, parallel=True)
def coordinate_descent(X, y, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
    # Inline the coordinate_descent logic here
    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y)

    _, n_features = X.shape
    beta = np.zeros(n_features)
    Xy = np.dot(X.T, y)  # Precompute X^T y

    for _ in range(max_iter):
        beta_old = beta.copy()

        for j in range(n_features):
            X_col_j = np.ascontiguousarray(X[:, j])
            residual = np.dot(X, beta) - X[:, j] * beta[j]
            residual = np.ascontiguousarray(residual)

            rho = np.dot(X_col_j, y - residual)

            # Update beta[j] using soft-thresholding
            if rho < -alpha * l1_ratio:
                beta[j] = (rho + alpha * l1_ratio) / (np.dot(X_col_j, X_col_j) + alpha * (1 - l1_ratio))
            elif rho > alpha * l1_ratio:
                beta[j] = (rho - alpha * l1_ratio) / (np.dot(X_col_j, X_col_j) + alpha * (1 - l1_ratio))
            else:
                beta[j] = 0.0

        if np.linalg.norm(beta - beta_old) < tol:
            break

    return beta

@timed_evaluation
@jit(cache=True, parallel=True)
def numba_elastic_net(X_train, y_train, X_test, y_test, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, k_folds=5):
    """Perform k-fold cross-validation for the Numba-accelerated Elastic Net."""
    n_samples = X_train.shape[0]
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  # Shuffle the data indices

    mse_scores = np.zeros(k_folds)  # To store MSE for each fold

    for fold in range(k_folds):
        # Train the model on the training set
        beta = coordinate_descent(X_train, y_train, alpha, l1_ratio, max_iter, tol)
        
        # Predict on validation set
        y_pred = np.dot(X_test, beta)

        # Calculate Mean Squared Error (MSE) for validation set
        mse = np.mean((y_test - y_pred) ** 2)
        mse_scores[fold] = mse

    # Return the average MSE across folds
    return beta, np.mean(mse_scores)

