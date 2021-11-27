import numpy as np
import os


def load_data(num_days_extended=0,
              labelled_path='./minder_utils/data/weekly_test/previous/npy/labelled',
              unlabelled_path='./minder_utils/data/weekly_test/previous/npy/unlabelled',
              flatten=True):
    '''
    This function is used to load the labelled data and unlabelled data.

    You can get this data by
        - weekly_loader, then the data will be saved into the subfolders in
            ./data/weekly_test/previous/npy
        - Alternatively, you can get theses data by Dataloader and Formatting.

    Parameters
    ----------
    labelled_path
    unlabelled_path
    valid_only

    Returns
    -------

    '''
    X = np.load(os.path.join(labelled_path, 'activity.npy'))
    y = np.load(os.path.join(labelled_path, 'label.npy'))
    p_ids = np.load(os.path.join(labelled_path, 'patient_id.npy'))
    unlabelled = np.load(os.path.join(unlabelled_path, 'activity.npy'))

    if flatten:
        indices = list(y[0][1: num_days_extended * 2 + 1]) + [-1, 1]
        X = X[np.isin(y, indices)].reshape(-1, 3, 8, 14)
        p_ids = p_ids[np.isin(y, indices)]
        y = y[np.isin(y, indices)]
        y[y > 0] = 1
        y[y < 0] = -1

        unlabelled = unlabelled.reshape(unlabelled.shape[0], 3, 8, 14)
        X.reshape(X.shape[0], 3, 8, 14)
    else:
        X = X.reshape(X.shape[0], X.shape[1], 3, 8, 14)

    return unlabelled, X, y, p_ids
