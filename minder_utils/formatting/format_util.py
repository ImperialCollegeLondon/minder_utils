import numpy as np
import os
from sklearn.preprocessing import Normalizer


def iter_dir(directory, endwith='.csv', split=True):
    """
    iterate csv files in a given directory
    :param directory: path to the folder
    :@param endwith:
    :return: list of file names end with csv
    """
    filenames = []
    for name in os.listdir(directory):
        if name.endswith(endwith):
            if split:
                filenames.append(name.split('.')[0])
            else:
                filenames.append(name)
    return filenames


def y_to_categorical(y, smooth=False, valid_only=False):
    if valid_only:
        mask = np.isin(y, [-1, 1])
        y = y[mask]
    positives = y > 0
    labels = np.zeros((y.shape[0], 2))
    if smooth:
        labels[:, 1][positives.reshape(-1, )] = y[y > 0]
        labels[:, 0][~positives.reshape(-1, )] = np.abs(y[y < 0])
    else:
        labels[:, 1][positives.reshape(-1, )] = 1
        labels[:, 0][~positives.reshape(-1, )] = 1
    if valid_only:
        return labels, mask
    else:
        return labels


def normalise(X, technique='l2'):
    assert technique in ['z-score', 'max-min', 'l2', 'l1', 'max', None], 'not implemented ...'
    if technique is None:
        pass
    elif technique == 'z-score':
        for i in range(X.shape[1]):
            data = X[:, i]
            std = 1 if np.std(data) == 0 else np.std(data)
            X[:, i] = (data - np.mean(data)) / std
    elif technique == 'max-min':
        for i in range(X.shape[1]):
            data = X[:, i]
            std = 1 if np.max(data) == np.min(data) else np.max(data) - np.min(data)
            X[:, i] = (data - np.min(data)) / std
    elif technique in ['l1', 'l2', 'max']:
        X = X.reshape(X.shape[0], -1)
        X = Normalizer(technique).fit_transform(X.transpose(1, 0)).transpose(0, 1)
        # for i in range(X.shape[2]):
        # data = X[:, i].reshape(-1, 1)
        # X[:, i] = Normalizer(technique).fit_transform(data).reshape(-1)
        #   X[:, :, i] = Normalizer(technique).fit_transform(X[:, :, i])
    return X


def format_mean_std(values):
    return str(np.mean(values))[:6] + " +/- " + str(np.std(values))[:6]


def flatten(x, last_axis=False):
    if last_axis:
        return x.reshape(x.shape[0], x.shape[1], -1)
    return x.reshape(x.shape[0], -1)


def l2_norm(x, epsilon=1e-10):
    return x / np.sqrt(max(np.sum(x ** 2), epsilon))
