import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class LogisticsRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self.w = None
        self.losses = []

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        l = (1 - y) * np.log(1 - y_hat + 10e-15) + y * np.log(y_hat + 10e-15)
        return -l.mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        self.w = np.zeros((d, 1), dtype=np.float64)
        
        for e in range(self.epoch):
            y_hat = self.predict(X)
            delta_y = (y_hat - y)
            gradient = delta_y.T @ X
            self.w = self.w - (1 / N) * self.lr * gradient.T
            
            l = self.loss_fn(y, y_hat)
            self.losses.append(l)

    def sigmoid(self, z: np.ndarray):
        return 1 / (1 + np.exp(-z))

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w
        y_hat = self.sigmoid(z)
        return y_hat

    def evaluate(self, y: np.ndarray, y_hat: np.ndarray) -> dict:
        y_pred = (y_hat >= 0.5).astype(int)
        return {
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0)
        }