import numpy as np
import torch
import torch.nn.functional as F


class Partial_Order_Loader:
    def __init__(self, data, y=None, shuffle=True, augmented_day=3, max_iter=None, normalise=True):
        self.data = data
        self.y = y
        self.augmented_day = augmented_day
        self.max_iter = max_iter
        self.iter_count = 0
        self.shuffle = shuffle
        self.normalise = normalise

    def normalisation(self, data):
        if self.normalise:
            data = torch.Tensor(data)
            return F.normalize(data.view(24, -1), dim=0).view(data.size()).detach().numpy()
        return data

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_count >= len(self):
            raise StopIteration
        if self.y is not None:
            data_idx = np.random.choice(len(self.data)) if self.shuffle else self.iter_count
            p_data = self.data[data_idx]
            p_label = self.y[data_idx]
            anchor = self.normalisation(p_data[np.isin(p_label, [-1, 1])])

            pre_idx = np.sort(p_label[p_label < 0])
            post_idx = np.sort(p_label[(p_label > 0) & (p_label != 1)])[::-1]
            pre_anchor = []
            post_anchor = []
            for i in range(self.augmented_day):
                pre_anchor.append(self.normalisation(p_data[p_label == pre_idx[i]]))
                post_anchor.append(self.normalisation(p_data[p_label == post_idx[i]]))
            # pre_anchor = np.concatenate(pre_anchor)
            # post_anchor = np.concatenate(post_anchor)
            self.iter_count += 1
        else:
            data_idx = np.random.choice(len(self.data) - self.augmented_day * 2) + self.augmented_day
            p_data = self.data[data_idx]
            anchor = self.normalisation(p_data)

            pre_anchor = []
            post_anchor = []
            for i in range(1, self.augmented_day + 1):
                pre_anchor.append(self.normalisation(p_data[data_idx - i]))
                post_anchor.append(self.normalisation(p_data[data_idx + i]))
            self.iter_count += 1

        pre_anchor, anchor, post_anchor = torch.Tensor(pre_anchor), torch.Tensor(anchor), torch.Tensor(post_anchor)

        return pre_anchor, anchor, post_anchor

    def __len__(self):
        if self.max_iter:
            return self.max_iter
        return len(self.data)
