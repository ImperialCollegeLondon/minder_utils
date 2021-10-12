import pandas as pd
import numpy as np
import datetime
# import torch
# import torch.nn.functional as F
from formatting.format_util import normalise as normalized
from formatting.label import label_dataframe


class Dataloader:
    """
    Load the formatted data
    """

    def __init__(self, csv_file='table.csv', max_days=3, label_data=False):
        """
        csv_file: string or dataframe, the resulting dataframe must contains
                    [id, time, ..., valid]
        max_days: the number of days to augment the labelled data
        """
        df = pd.read_csv(csv_file) if type(csv_file) == str else csv_file
        df.time = pd.to_datetime(df.time)
        df['Date'] = df.time.dt.date

        if label_data:
            df = label_dataframe(df)
            self.labelled_df = df[~df.valid.isna()]
            self.labelled_df.set_index(['id', 'valid', 'Date'], inplace=True)
            if len(self.labelled_df) > 0:
                self.true_p_ids = self.labelled_df.loc[:, True, :].index.get_level_values(0).unique()
                self.false_p_ids = self.labelled_df.loc[:, False, :].index.get_level_values(0).unique()
            else:
                print('no data is labelled')

        df.set_index(['id', 'Date'], inplace=True)
        self.df = df
        self.max_days = max_days

        self.transfer_sensors = ['back door', 'bathroom1', 'bedroom1', 'dining room',
                                 'fridge door', 'front door', 'hallway', 'kettle', 'kitchen',
                                 'living room', 'lounge', 'microwave', 'study', 'toaster']
        self.select_sensors = ['WC1', 'back door', 'bathroom1', 'bedroom1', 'cellar',
                               'conservatory', 'dining room', 'fridge door', 'front door',
                               'hallway', 'iron', 'kettle', 'kitchen', 'living room',
                               'lounge', 'main door', 'microwave', 'multi', 'office']

    def __len__(self):
        return int(len(self.labelled_df) / 24)

    def get(self, valid=None):
        if valid is None:
            valid = bool(np.random.randint(2))
        outputs = []
        # get p ids
        p_ids = self.true_p_ids if valid else self.false_p_ids
        idx = np.random.randint(len(p_ids))
        # get data of patient
        data = self.labelled_df.loc[:, valid, :].loc[p_ids[idx]]
        dates = np.unique(data.index.values)
        # get date of patient
        date = pd.to_datetime(dates[np.random.randint(len(dates))])
        # validated data
        data = data.loc[date, 'Back Door': 'Toaster'].to_numpy().reshape(3, 8, -1)
        outputs.append(data)
        for i in range(1, self.max_days):
            date = date - datetime.timedelta(1)
            try:
                outputs.append(self.df.loc[(p_ids[idx], date), 'Back Door': 'Toaster'].to_numpy().reshape(3, 8, -1))
            except KeyError:
                break
        outputs = np.array(outputs)

        label = np.array([int(valid)])

        return outputs, label

    def iterate_data(self, p_ids=None, normalise=True):
        # get p ids
        p_ids = self.labelled_df.index.get_level_values(0).unique() if p_ids is None else p_ids
        for idx in range(len(p_ids)):
            # get data of patient
            data = self.labelled_df.loc[p_ids[idx]]
            for valid in data.index.get_level_values(0).unique():
                dates = data.loc[valid].index.get_level_values(0).unique()
                for date in dates:
                    outputs = []
                    labels = []
                    patient_ids = []
                    # validated data
                    p_date = date
                    p_data = data.loc[(valid, p_date), self.select_sensors].to_numpy().reshape(3, 8, -1)
                    if normalise:
                        p_data = normalized(np.array(p_data)).reshape(3, 8, -1)
                        # p_data = normalized(np.array(p_data))
                    outputs.append(p_data)
                    labels.append(int(valid) if valid else -1)
                    patient_ids.append(p_ids[idx])
                    for i in range(1, self.max_days + 1):
                        for symbol in [-1, 1]:
                            f_date = p_date - datetime.timedelta(i) * symbol
                            try:
                                p_data = self.df.loc[(p_ids[idx], f_date), self.select_sensors].to_numpy()
                                if normalise:
                                    p_data = normalized(np.array(p_data)).reshape(3, 8, -1)
                                    # p_data = normalized(np.array(p_data), axis=-1).reshape(3, 8, -1)
                                outputs.append(p_data)
                                labels.append(self.laplace_smooth(i) * symbol)
                                patient_ids.append(p_ids[idx])
                            except KeyError:
                                break

                    outputs = np.array(outputs)
                    label = np.array(labels)
                    patient_ids = np.array(patient_ids)

                    yield outputs, label, patient_ids

    def get_unlabelled_data(self, normalise=True, date=None):
        # get p ids
        df = self.df.reset_index().set_index(['id', 'Date'])
        if date is not None:
            df = df[df.index.get_level_values(1) > date]
        p_ids = df.index.get_level_values(0).unique()
        outputs = []
        outputs_p_ids = []
        outputs_dates = []
        for idx in range(len(p_ids)):
            # get data of patient
            data = df.loc[p_ids[idx]]
            dates = data.index.get_level_values(0).unique()
            for date in dates:
                # validated data
                p_data = data.loc[date, self.select_sensors].to_numpy()
                if normalise:
                    # p_data = torch.Tensor(np.array(p_data))
                    # p_data = F.normalize(p_data, p=2, dim=-1)
                    p_data = normalized(np.array(p_data)).reshape(3, 8, -1)
                    # p_data = np.array(p_data)
                    # p_data = normalize(p_data, axis=2)
                outputs.append(p_data)
                outputs_p_ids.append(p_ids[idx])
                outputs_dates.append(date)
        if normalise:
            # return torch.stack(outputs)
            return np.stack(outputs), np.array(outputs_p_ids), np.array(outputs_dates)
        # return torch.Tensor(np.array(outputs))
        return np.array(outputs), np.array(outputs_p_ids), np.array(outputs_dates)

    @staticmethod
    def laplace_smooth(i, lam=3, denominator=1):
        return np.exp(- np.abs(i) / lam) / denominator
