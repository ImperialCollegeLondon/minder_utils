import pandas as pd
import numpy as np
import datetime
# import torch
# import torch.nn.functional as F
from ..formatting.format_util import normalise as normalized
from ..formatting.label import label_dataframe
from minder_utils.formatting.standardisation import standardise_physiological_environmental, standardise_activity_data


class Dataloader:
    """
    Categorise the data into labelled & unlabelled data.
    This dataloader should be used combined with minder_utils.formatting.Formatting.

    After initialising ```formater = Formatting()```
    Parameters:
        - activity: activity data, ```formater.activity```
        - physiological: physiological data, ```formater.physiological```
        - environmental: environmental data, ```formater.environmental```
        - max_days: Default 3. How many consecutive days to extended as UTI, if ```max_days = n```, ```n``` days before & after
            the validated date will be labelled as UTI
        - label_data: Default False. label the data or not. If False, ```get_labelled_data()``` cannot be used.
    """

    def __init__(self, activity, physiological=None, environmental=None,  max_days=3, label_data=False):
        """
        csv_file: string or dataframe, the resulting dataframe must contains
                    [id, time, ..., valid]
        max_days: the number of days to augment the labelled data
        """
        activity = pd.read_csv(activity) if type(activity) == str else activity
        shared_id = None
        for data in [activity, physiological, environmental]:
            if data is None:
                continue
            shared_id = set(data.id.unique()) if shared_id is None else shared_id.intersection(set(data.id.unique()))
        activity = activity[activity.id.isin(shared_id)]
        activity = standardise_activity_data(activity)
        activity.time = pd.to_datetime(activity.time)
        activity.loc[:, 'Date'] = activity.time.dt.date
        date_range = pd.date_range(activity.Date.min(), activity.Date.max())

        self.physiological = standardise_physiological_environmental(physiological, date_range, shared_id)\
            .set_index(['id', 'time']) if physiological is not None else physiological
        self.environmental = standardise_physiological_environmental(environmental, date_range, shared_id)\
            .set_index(['id', 'time']) if environmental is not None else environmental

        if label_data:
            activity = label_dataframe(activity)
            self.labelled_df = activity[~activity.valid.isna()]
            self.labelled_df.set_index(['id', 'valid', 'Date'], inplace=True)
            if len(self.labelled_df) > 0:
                self.true_p_ids = self.labelled_df.loc[:, True, :].index.get_level_values(0).unique()
                self.false_p_ids = self.labelled_df.loc[:, False, :].index.get_level_values(0).unique()
            else:
                print('no data is labelled')

        activity.set_index(['id', 'Date'], inplace=True)
        self.activity = activity
        self.max_days = max_days

        self.transfer_sensors = ['back door', 'bathroom1', 'bedroom1', 'dining room',
                                 'fridge door', 'front door', 'hallway', 'kettle', 'kitchen',
                                 'living room', 'lounge', 'microwave', 'study', 'toaster']
        self.select_sensors = ['WC1', 'back door', 'bathroom1', 'bedroom1',
                               'conservatory', 'dining room', 'fridge door', 'front door',
                               'hallway', 'kettle', 'kitchen', 'living room',
                               'lounge', 'main door', 'microwave', 'office']

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
                outputs.append(self.activity.loc[(p_ids[idx], date), 'Back Door': 'Toaster'].to_numpy().reshape(3, 8, -1))
            except KeyError:
                break
        outputs = np.array(outputs)

        label = np.array([int(valid)])

        return outputs, label

    def get_labelled_data(self, normalise=True):
        # get p ids
        p_ids = self.labelled_df.index.get_level_values(0).unique()
        outputs = []
        labels = []
        patient_ids = []
        phy_data = []
        env_data = []
        for idx in range(len(p_ids)):
            # get data of patient
            data = self.labelled_df.loc[p_ids[idx]]
            for valid in data.index.get_level_values(0).unique():
                dates = data.loc[valid].index.get_level_values(0).unique()
                for date in dates:
                    # validated data
                    p_date = date
                    p_data = data.loc[(valid, p_date), self.select_sensors].to_numpy().reshape(3, 8, -1)
                    if normalise:
                        p_data = normalized(np.array(p_data)).reshape(3, 8, -1)
                        # p_data = normalized(np.array(p_data))
                    outputs.append(p_data)
                    phy_data.append(self.get_data(self.physiological, p_ids[idx], p_date))
                    env_data.append(self.get_data(self.environmental, p_ids[idx], p_date))
                    labels.append(int(valid) if valid else -1)
                    patient_ids.append(p_ids[idx])
                    for i in range(1, self.max_days + 1):
                        for symbol in [-1, 1]:
                            f_date = p_date - datetime.timedelta(i) * symbol
                            try:
                                p_data = self.activity.loc[(p_ids[idx], f_date), self.select_sensors].to_numpy()
                                if normalise:
                                    p_data = normalized(np.array(p_data)).reshape(3, 8, -1)
                                    #p_data = normalized(np.array(p_data), axis=-1).reshape(3, 8, -1)
                                outputs.append(p_data)
                                phy_data.append(self.get_data(self.physiological, p_ids[idx], f_date))
                                env_data.append(self.get_data(self.environmental, p_ids[idx], f_date))
                                labels.append(self.laplace_smooth(i) * symbol)
                                patient_ids.append(p_ids[idx])
                            except KeyError:
                                break

        outputs = np.array(outputs)
        phy_data = np.array(phy_data)
        env_data = np.array(env_data)
        label = np.array(labels)
        patient_ids = np.array(patient_ids)

        return outputs, phy_data, env_data, label, patient_ids

    def get_unlabelled_data(self, normalise=True, date=None):
        # get p ids
        df = self.activity.reset_index().set_index(['id', 'Date'])
        if date is not None:
            df = df[df.index.get_level_values(1) > date]
        p_ids = df.index.get_level_values(0).unique()
        outputs = []
        phy_data, env_data = [], []
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
                phy_data.append(self.get_data(self.physiological, p_ids[idx], date))
                env_data.append(self.get_data(self.environmental, p_ids[idx], date))
                outputs_p_ids.append(p_ids[idx])
                outputs_dates.append(date)
        return np.array(outputs), np.array(phy_data), np.array(env_data), \
               np.array(outputs_p_ids), np.array(outputs_dates)

    @staticmethod
    def laplace_smooth(i, lam=3, denominator=1):
        return np.exp(- np.abs(i) / lam) / denominator

    @staticmethod
    def get_data(df, p_id, date):
        if df is None:
            return
        return df.loc[(p_id, date)].sort_values('location')['value'].to_numpy()
