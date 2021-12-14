import numpy as np


class ZScore:
    def __init__(self):
        return

    def fit(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)
        return

    def decision_function(self, X):
        return np.mean((X - self.mean) / self.std)
