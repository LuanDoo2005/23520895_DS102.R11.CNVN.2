import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LinearRegression:
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.beta = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_mat = np.array(X, dtype=np.float64)
        y_vec = np.array(y, dtype=np.float64).reshape(-1, 1)

        if self.fit_intercept:
            bias = np.ones((X_mat.shape[0], 1), dtype=np.float64)
            X_mat = np.hstack([bias, X_mat])

        self.beta = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_vec
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_mat = np.array(X, dtype=np.float64)
        if self.fit_intercept:
            bias = np.ones((X_mat.shape[0], 1), dtype=np.float64)
            X_mat = np.hstack([bias, X_mat])

        return X_mat @ self.beta

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true = np.array(y_true).reshape(-1, 1)
        y_pred = np.array(y_pred).reshape(-1, 1)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }
