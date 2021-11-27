import numpy as np
import os


def process_data(data, unlabelled_data, num_days_extended=0, flatten=True):
    '''
    This function is to process the data from dataloader and make it easy for
    the models to use
    Parameters
    ----------
    data: dict, dictionary contains activity, p_ids, uti_labels
    unlabelled_data: numpy array, unlabelled data
    num_days_extended: int, How many consecutive days you want to extend the labelled data.
        The data will be extended by 2 * num_days_extended (n days before and after)
    flatten: bool, flatten the activity data or not.

    Returns list contains unlabelled_data, X, y, p_ids
    -------

    '''
    X, y, p_ids = data['activity'], data['uti_labels'], data['p_ids']
    X_truncated = []
    y_truncated = []
    p_ids_truncated = []
    for idx in range(len(X)):
        truncated_len = 1 + num_days_extended * 2
        if len(X[idx]) >= truncated_len:
            X_truncated.append(X[idx][: truncated_len])
            y_truncated.append(y[idx][: truncated_len])
            p_ids_truncated.append(p_ids[idx][: truncated_len])
    X, y, p_ids = np.array(X_truncated), np.array(y_truncated), np.array(p_ids_truncated)

    if flatten:
        X = X.reshape(X.shape[0], 3, 8, 14)
        y = y.reshape(y.shape[0], -1)
        y[y > 0] = 1
        y[y < 0] = -1
    else:
        X = X.reshape(X.shape[0], X.shape[1], 3, 8, 14)

    unlabelled_data = unlabelled_data.reshape(unlabelled_data.shape[0], 3, 8, 14)
    return unlabelled_data, X, y, p_ids
