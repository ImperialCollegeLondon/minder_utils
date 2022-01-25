import numpy as np


class ZScore:
    def __init__(self):
        return

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = np.nan
        return

    def decision_function(self, X):
        return np.nanmean((X - self.mean) / self.std)
