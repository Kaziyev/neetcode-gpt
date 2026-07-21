import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        a = np.log(1 - y_pred)
        b = 1 - y_true
        c = a * b
        d = y_true * np.log(y_pred)
        e = -np.mean(c + d)
        return round(float(e), 4)
        

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return round(float(loss), 4)