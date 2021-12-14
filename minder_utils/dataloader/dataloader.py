import pandas as pd
import numpy as np
import warnings
import datetime
from minder_utils.util import load_save
from minder_utils.formatting.format_util import normalise as normalized
from minder_utils.formatting.label import label_dataframe
from minder_utils.formatting.standardisation import standardise_physiological_environmental, standardise_activity_data
from minder_utils.configurations import config


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

    def __init__(self, activity, physiological=None, environmental=None, max_days=3, label_data=False):
        if activity is None:
            warnings.warn('Activity data is None, this class can be only used to load the processed data')
            return
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

        self.physiological = standardise_physiological_environmental(physiological, date_range, shared_id) \
            if physiological is not None else physiological
        self.environmental = standardise_physiological_environmental(environmental, date_range, shared_id) \
            if environmental is not None else environmental

        for datatype in ['environmental', 'physiological']:
            config[datatype]['sort_dict'] = dict(
                zip(config[datatype]['sensors'], range(len(config[datatype]['sensors']))))

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
        self.select_sensors = config['activity']['sensors']

    def __len__(self):
        return int(len(self.labelled_df) / 24)

    @property
    @load_save(**config['labelled_data']['save'])
    def labelled_data(self):
        activity_data, physiological_data, environmental_data, patient_ids, uti_labels = \
            self.get_labelled_data(normalise=False)
        return {
            'activity': activity_data,
            'phy': physiological_data,
            'env': environmental_data,
            'p_ids': patient_ids,
            'uti_labels': uti_labels
        }

    @property
    @load_save(**config['unlabelled_data']['save'])
    def unlabelled_data(self):
        activity_data, physiological_data, environmental_data, patient_ids, dates = \
            self.get_unlabelled_data(normalise=False)
        return {
            'activity': activity_data,
            'phy': physiological_data,
            'env': environmental_data,
            'p_ids': patient_ids,
            'dates': dates
        }

    def get_labelled_data(self, normalise=False):
        # get p ids
        p_ids = self.labelled_df.index.get_level_values(0).unique()
        activity_data, uti_labels, patient_ids, physiological_data, environmental_data = [], [], [], [], []
        for idx in range(len(p_ids)):
            # get data of patient
            data = self.labelled_df.loc[p_ids[idx]]
            for valid in data.index.get_level_values(0).unique():
                dates = data.loc[valid].index.get_level_values(0).unique()
                for date in dates:
                    # validated data
                    act_data, labels, patient, phy_data, env_data = [], [], [], [], []
                    p_date = date
                    p_data = data.loc[(valid, p_date), self.select_sensors].to_numpy()
                    if normalise:
                        p_data = normalized(np.array(p_data)).reshape(3, 8, -1)
                        # p_data = normalized(np.array(p_data))
                    act_data.append(p_data)
                    phy_data.append(self.get_data(self.physiological, p_ids[idx], p_date, 'physiological'))
                    env_data.append(self.get_data(self.environmental, p_ids[idx], p_date, 'environmental'))
                    labels.append(int(valid) if valid else -1)
                    patient.append(p_ids[idx])
                    for i in range(1, self.max_days + 1):
                        for symbol in [-1, 1]:
                            f_date = p_date - datetime.timedelta(i) * symbol
                            try:
                                p_data = self.activity.loc[(p_ids[idx], f_date), self.select_sensors].to_numpy()
                                if normalise:
                                    p_data = normalized(np.array(p_data)).reshape(3, 8, -1)
                                    # p_data = normalized(np.array(p_data), axis=-1).reshape(3, 8, -1)
                                act_data.append(p_data)
                                phy_data.append(self.get_data(self.physiological, p_ids[idx], f_date, 'physiological'))
                                env_data.append(self.get_data(self.environmental, p_ids[idx], f_date, 'environmental'))
                                labels.append(self.laplace_smooth(i) * symbol)
                                patient.append(p_ids[idx])
                            except KeyError:
                                break
                    activity_data.append(act_data)
                    uti_labels.append(labels)
                    patient_ids.append(patient)
                    physiological_data.append(phy_data)
                    environmental_data.append(env_data)

        activity_data = np.array(activity_data)
        uti_labels = np.array(uti_labels)
        patient_ids = np.array(patient_ids)
        physiological_data = np.array(physiological_data)
        environmental_data = np.array(environmental_data)

        return activity_data, physiological_data, environmental_data, patient_ids, uti_labels

    def get_unlabelled_data(self, normalise=False, date='2021-03-01'):
        '''
        Get the unlabelled data,
        Parameters
        ----------
        normalise: bool, normalise the data or not
        date: str, only return the data later than the date provided. By default,
            it will not return the tihm unlabelled

        Returns activity, physiological, environmental data, patient ids, dates
        -------

        '''
        # May need to change the for loop to dataframe operations
        # df = self.activity.reset_index().set_index(['id', 'Date'])
        # phy_df = self.physiological.reset_index()
        # phy_df = phy_df.pivot_table(index=['id', 'time'], columns='location',
        #                             values='value').reset_index().rename(columns={'time': 'Date'})
        # indices = df.reset_index()[['id', 'Date']].drop_duplicates()
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
                phy_data.append(self.get_data(self.physiological, p_ids[idx], date, 'physiological'))
                env_data.append(self.get_data(self.environmental, p_ids[idx], date, 'environmental'))
                outputs_p_ids.append(p_ids[idx])
                outputs_dates.append(date)
        return np.array(outputs), np.array(phy_data), np.array(env_data), \
               np.array(outputs_p_ids), np.array(outputs_dates)

    @staticmethod
    def laplace_smooth(i, lam=3, denominator=1):
        return np.exp(- np.abs(i) / lam) / denominator

    @staticmethod
    def get_data(df, p_id, date, datatype):
        if df is None:
            return
        return df.loc[(p_id, date, config[datatype]['sensors'])] \
            .sort_values('location', key=lambda x: x.map(config[datatype]['sort_dict']))['value'].to_numpy()
