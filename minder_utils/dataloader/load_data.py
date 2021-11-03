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
    loader = Weekly_dataloader()

    unlabelled, activity_data, y, p_ids = [], [], [], []
    if load_TIHM:
        unlabelled.append(np.load(os.path.join('./data/TIHM/', 'unlabelled.npy'), allow_pickle=True)[0].numpy())
        activity_data.append(np.load('./data/TIHM/data.npy'))
        y.append(np.load('./data/TIHM/label.npy'))
        p_ids.append(np.load('./data/TIHM/p_ids.npy'))
    if load_DRI:
        dri_unlabelled = np.load(os.path.join(loader.previous_unlabelled_data, 'activity.npy'))
        unlabelled.append(dri_unlabelled.reshape(dri_unlabelled.shape[0], 3, 8, -1))
        act_data = np.load('./data/weekly_test/previous/npy/labelled/activity.npy')
        act_data = act_data.reshape(act_data.shape[0], act_data.shape[1], 3, 8, 14)
        activity_data.append(act_data)

        y.append(np.load(os.path.join(loader.previous_labelled_data, 'label.npy')))
        p_ids.append(np.load(os.path.join(loader.previous_labelled_data, 'patient_id.npy')))
    unlabelled = np.concatenate(unlabelled)
    X = np.concatenate(activity_data)
    y = np.concatenate(y)
    p_ids = np.concatenate(p_ids)

    if valid_only:
        X = X[:, 0]
        y = y[:, 0]
        p_ids = p_ids[:, 0]

    return unlabelled, X, y, p_ids