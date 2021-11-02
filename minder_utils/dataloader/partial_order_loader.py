import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class Partial_Order_Loader:
    def __init__(self, data, y=None, augmented_day=3, max_iter=10000):
        assert augmented_day >= 2, 'less than two days'
        self.data = data
        self.y = y
        self.augmented_day = augmented_day
        self.max_iter = max_iter



class Train_loader:
    def __init__(self, normalisation, augmented_day=3, max_iter=10000):
        assert augmented_day >= 2, 'less than two days'
        self.unlabelled_data = np.load('./data/raw_data/unlabelled.npy')
        # self.unlabelled_data = self.unlabelled_data.reshape(self.unlabelled_data.shape[0], -1)
        self.unlabelled_data = normalise(self.unlabelled_data, normalisation)
        self.patient_ids = np.load('./data/raw_data/unlabelled_ids.npy')
        self.dates = np.load('./data/raw_data/unlabelled_dates.npy')
        self.augmented_day = augmented_day
        self.max_iter = max_iter
        self.iter_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_count > self.max_iter:
            raise StopIteration
        p_id = np.random.choice(self.patient_ids)
        p_data = self.unlabelled_data[self.patient_ids == p_id]
        dates = self.dates[self.patient_ids == p_id]
        try:
            date_idx = np.random.choice(len(dates) - self.augmented_day * 2) + self.augmented_day
        except ValueError:
            return self.__next__()

        anchor = p_data[date_idx]
        pre_anchor = []
        post_anchor = []
        for i in range(1, self.augmented_day + 1):
            pre_anchor.append(p_data[date_idx - i])
            post_anchor.append(p_data[date_idx + i])
        self.iter_count += 1
        return torch.Tensor(pre_anchor), torch.Tensor(post_anchor), torch.Tensor(anchor)

    def __len__(self):
        if self.max_iter is not None:
            return self.max_iter
        return len(self.unlabelled_data)


class Train_loader_label:
    def __init__(self, normalisation, augmented_day=3):
        assert augmented_day >= 2, 'less than two days'
        self.unlabelled_data = np.load('./data/raw_data/X.npy')
        # self.unlabelled_data = self.unlabelled_data.reshape(self.unlabelled_data.shape[0], -1)
        self.unlabelled_data = normalise(self.unlabelled_data, normalisation)
        self.patient_ids = np.load('./data/raw_data/patient_ids.npy')
        self.label = np.load('./data/raw_data/y.npy')
        self.augmented_day = augmented_day
        self.indices = np.where(np.isin(self.label, [-1, 1]))[0]
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            date_idx = self.indices[self.idx]
        except IndexError:
            raise StopIteration
        anchor = self.unlabelled_data[date_idx]
        pre_anchor = []
        post_anchor = []
        for i in range(1, self.augmented_day + 1):
            pre_anchor.append(self.unlabelled_data[date_idx + i])
            post_anchor.append(self.unlabelled_data[date_idx + i + 6])
        self.idx += 1
        return torch.Tensor(pre_anchor), torch.Tensor(post_anchor), torch.Tensor(anchor)

    def __len__(self):
        return len(self.unlabelled_data)


def get_test_loader(normalisation, valid_only=True):
    X = np.load('./data/raw_data/X.npy')
    y = np.load('./data/raw_data/y.npy')
    patient_ids = np.load('./data/raw_data/patient_ids.npy')

    if valid_only:
        idx = np.isin(y, [-1, 1])
        X, y = X[idx], y[idx]
        np.save('./data/DRI/patient_ids.npy', patient_ids[idx])
    # X = X.reshape(X.shape[0], -1)
    X = normalise(X, normalisation)
    y[y == -1] = 0
    test_data = TensorDataset(torch.Tensor(X), torch.Tensor(y))
    return DataLoader(test_data, batch_size=64, shuffle=False)


def split_data(valid_only=True, seed=0):
    X = np.load('./data/DRI/x.npy')
    y = np.load('./data/DRI/y.npy')
    patient_ids = np.load('./data/DRI/patient_ids.npy')
    train_ids, test_ids = train_test_split(np.unique(patient_ids), test_size=0.33, random_state=seed)
    test_y = y[np.isin(patient_ids, test_ids)]
    if valid_only:
        test_filter = np.isin(test_y, [0, 1])
    else:
        test_filter = np.isin(test_y, np.unique(test_y))
    return X[np.isin(patient_ids, train_ids)], y[np.isin(patient_ids, train_ids)], \
           X[np.isin(patient_ids, test_ids)][test_filter], y[np.isin(patient_ids, test_ids)][test_filter]
