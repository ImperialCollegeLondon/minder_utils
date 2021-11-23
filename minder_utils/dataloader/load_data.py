from minder_utils.formatting.format_util import flatten
from minder_utils.scripts.weekly_loader import Weekly_dataloader
import numpy as np
import os


def load_data(load_TIHM=True, load_DRI=True, valid_only=False):
    '''
    This function is used to load the labelled data from TIHM (need to be
    downloaded and pre-processed separately) and DRI (need to be downloaded and
    pre-processed by weekly loader).
    Args:
        load_TIHM: Concatenate TIHM data or not
        valid_only:

    Returns: unlabelled data, labelled data, label, patient ids of labelled data

    '''
    path = './minder_utils/data/weekly_test/previous/npy/labelled'
    unlabelled_path = './minder_utils/data/weekly_test/previous/npy/unlabelled'
    X = np.load(os.path.join(path, 'activity.npy'))
    y = np.load(os.path.join(path, 'label.npy'))
    p_ids = np.load(os.path.join(path, 'patient_id.npy'))
    unlabelled = np.load(os.path.join(unlabelled_path, 'activity.npy'))

    if valid_only:
        X = X[:, 0]
        y = y[:, 0]
        p_ids = p_ids[:, 0]

    return unlabelled, X, y, p_ids