from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from ..formatting.format_util import y_to_categorical


def get_scores(y_true, y_pred):
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        return None, None, None, None
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return sensitivity, specificity, acc, f1


def split_by_ids(X, y, patient_ids, cat=True, valid_only=True, stratify=True, seed=0):
    y[y == 0] = -1
    y = y.reshape(-1, )
    patient_ids = patient_ids.reshape(-1, )
    # make sure the train and test set got both positive and negative patients
    y_p_id = []
    for p_id in np.unique(patient_ids):
        _y = np.unique(y[patient_ids == p_id])
        rng = np.random.default_rng(seed)
        y_p_id.append(int(_y[0]) if len(_y) < 2 else rng.integers(0,2))
        seed += 1
    y_p_id = np.array(y_p_id)
    y_p_id[y_p_id < 0] = 0

    train_ids, test_ids = train_test_split(np.unique(patient_ids), test_size=0.33, random_state=seed, stratify=y_p_id if stratify else None)
    test_y = y[np.isin(patient_ids, test_ids)]
    if valid_only:
        test_filter = np.isin(test_y, [-1, 1])
    else:
        test_filter = np.isin(test_y, np.unique(test_y))
    if cat:
        return X[np.isin(patient_ids, train_ids)], y_to_categorical(y[np.isin(patient_ids, train_ids)]), \
               X[np.isin(patient_ids, test_ids)][test_filter], y_to_categorical(y[np.isin(patient_ids, test_ids)][
                   test_filter])
    return X[np.isin(patient_ids, train_ids)], y[np.isin(patient_ids, train_ids)], \
           X[np.isin(patient_ids, test_ids)], y[np.isin(patient_ids, test_ids)]



class StratifiedKFoldPids:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        return


    def get_n_splits(self):
        return self.n_splits


    def split_by_ids(self, y, pids, seed=0):
        y[y == 0] = -1
        y = y.reshape(-1, )
        pids = pids.reshape(-1, )
        # make sure the train and test set got both positive and negative patients
        y_p_id = []
        for p_id in np.unique(pids):
            _y = np.unique(y[pids == p_id])
            rng = np.random.default_rng(seed)
            y_p_id.append(int(_y[0]) if len(_y) < 2 else rng.integers(0,2))
            seed += 1
        y_p_id = np.array(y_p_id)
        y_p_id[y_p_id < 0] = 0
        splitter = StratifiedKFold(n_splits=self.n_splits, 
                                    shuffle=self.shuffle, 
                                    random_state=seed if self.shuffle else None)
        splits = list(splitter.split(np.unique(pids), y=y_p_id))
        return [[np.unique(pids)[train_idx], np.unique(pids)[test_idx]] for train_idx, test_idx in splits]


    def split(self, X, y, pids):
        rng = np.random.default_rng(self.random_state)
        seed = rng.integers(0,1e6)
        list_of_splits_pids = self.split_by_ids(y=y, pids=pids, seed=seed)
        list_of_splits = []
        for train_pids, test_pids in list_of_splits_pids:
            
            train_idx_new = np.arange(len(pids))[np.isin(pids, train_pids)]
            test_idx_new = np.arange(len(pids))[np.isin(pids, test_pids)]

            list_of_splits.append([
                train_idx_new,
                test_idx_new,
            ])
            
        return list_of_splits
