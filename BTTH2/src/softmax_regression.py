import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class SoftmaxRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self.w = None
        self.losses = []

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        l = y * np.log(y_hat + 10e-15)
        return -l.sum(axis=1).mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        C = y.shape[1]
        self.w = np.zeros((d, C), dtype=np.float64)
        
        for e in range(self.epoch):
            y_hat = self.predict(X)
            delta_y = (y_hat - y)
            gradient = delta_y.T @ X
            self.w = self.w - (1 / N) * self.lr * gradient.T
            
            l = self.loss_fn(y, y_hat)
            self.losses.append(l)

    def softmax(self, z: np.ndarray):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w
        y_hat = self.softmax(z)
        return y_hat

    def evaluate(self, y: np.ndarray, y_hat: np.ndarray) -> dict:
        y_true = np.argmax(y, axis=1) if y.ndim > 1 else y
        y_pred = np.argmax(y_hat, axis=1) if y_hat.ndim > 1 else y_hat
        return {
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)
        }