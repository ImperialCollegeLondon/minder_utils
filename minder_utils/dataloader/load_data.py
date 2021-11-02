from minder_utils.formatting.format_util import flatten
from minder_utils.scripts.weekly_loader import Weekly_dataloader
import numpy as np
import os


def load_data(load_TIHM=True):
    '''
    This function is used to load the labelled data from TIHM (need to be
    downloaded and pre-processed separately) and DRI (need to be downloaded and
    pre-processed by weekly loader).
    Args:
        load_TIHM: Concatenate TIHM data or not

    Returns: unlabelled data, labelled data, label, patient ids of labelled data

    '''
    loader = Weekly_dataloader()

    unlabelled = np.concatenate([
        np.load('./data/weekly_test/previous/npy/labelled/simclr.npy'),
        np.load('./data/TIHM/simclr_data.npy')])

    X = np.concatenate([
        flatten(np.load('./data/weekly_test/previous/npy/labelled/activity.npy')),
        flatten(np.concatenate(np.load('./data/TIHM/data.npy')))]) if load_TIHM else \
        flatten(np.load('./data/weekly_test/previous/npy/labelled/activity.npy'))

    y = np.concatenate([
        np.load(os.path.join(loader.previous_labelled_data, 'label.npy')),
        np.concatenate(np.load('./data/TIHM/label.npy'))
    ]) if load_TIHM else \
        np.load(os.path.join(loader.previous_labelled_data, 'label.npy'))

    p_ids = np.concatenate([
        np.load(os.path.join(loader.previous_labelled_data, 'patient_id.npy')),
        np.concatenate(np.load('./data/TIHM/p_ids.npy'))
    ]) if load_TIHM else \
        np.load(os.path.join(loader.previous_labelled_data, 'patient_id.npy'))

    return unlabelled, X, y, p_ids