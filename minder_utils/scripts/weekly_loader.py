from minder_utils.formatting.format_util import iter_dir
import os
import datetime as DT
from minder_utils.download.download import Downloader
from minder_utils.formatting.formatting import Formatting
from minder_utils.dataloader import Dataloader
import numpy as np
from minder_utils.util.util import save_mkdir, delete_dir
import json
from minder_utils.settings import dates_save, date_backup
import pandas as pd
from minder_utils.configurations import dates_path
from minder_utils.configurations import config


class Weekly_dataloader:
    """
    Support UTI only
    This class will
        - download all previous data if it have not been downloaded before
            - will be saved as labelled data and unlabelled data
        - download the latest weekly data
        - reformat all the data into following (N is the number of samples)
            - activity data: N * 3 * 8 * 20
            - environmental data: N * 19
            - physiological data: N * 12
    """

    def __init__(self, categories=None, save_dir=os.path.join('./data', 'weekly_test'), num_days_extended=3):
        '''

        @param data_type: activity, environmental, physiological
        @param num_days_extended: for uti only, how many consecutive days to be labelled
        '''
        self.default_categories = ['activity', 'environmental', 'physiological']
        self.categories = self.default_categories if categories is None else categories
        assert all(data_type in self.default_categories for data_type in self.categories), 'available categories: ' \
                                                                                           'activity, environmental, ' \
                                                                                           'physiological'
        self.num_days_extended = num_days_extended
        self.downloader = Downloader()
        self.default_dir = save_dir
        save_mkdir(self.default_dir)

    @property
    def previous_labelled_data(self):
        return os.path.join(self.default_dir, 'previous', 'npy', 'labelled')

    @property
    def previous_unlabelled_data(self):
        return os.path.join(self.default_dir, 'previous', 'npy', 'unlabelled')

    @property
    def current_data(self):
        return os.path.join(self.default_dir, 'current', 'npy', 'unlabelled')

    @property
    def current_csv_data(self):
        return os.path.join(self.default_dir, 'current', 'csv')

    @property
    def previous_csv_data(self):
        return os.path.join(self.default_dir, 'previous', 'csv')

    @property
    def gap_csv_data(self):
        return os.path.join(self.default_dir, 'gap', 'csv')

    def initialise(self):
        dates_save(refresh=True)
        for folder in ['current', 'previous']:
            delete_dir(os.path.join(self.default_dir, folder, 'csv'))
            save_mkdir(os.path.join(self.default_dir, folder, 'csv'))
            delete_dir(os.path.join(self.default_dir, folder, 'npy'))
            save_mkdir(os.path.join(self.default_dir, folder, 'npy'))
            self.download(folder, include_devices=True)
            self.format(folder)

    def check_exist(self, path):
        check_list = {
            '.csv': {'activity': ['raw_door_sensor', 'raw_appliance_use', 'raw_activity_pir', 'device_types']},
            '.npy': {'activity': {'current': ['unlabelled', 'patient_id', 'dates'],
                                  'previous': ['unlabelled', 'patient_id', 'dates', 'X', 'y']}},
        }
        folder_type = 'previous' if 'previous' in path else 'current'
        reformat_flag = False
        for data_type in ['activity']:
            # Check the csv file
            if not set([ele + '.csv' for ele in check_list['.csv'][data_type]]) \
                   <= set(iter_dir(os.path.join(path, 'csv'), '.csv', False)):
                print(data_type, folder_type, 'raw data does not exist, start to download')
                self.download(folder_type, data_type)
                reformat_flag = True
            else:
                print(data_type, folder_type, 'is already downloaded')

            # Check the npy file
            if not set([ele + '.npy' for ele in check_list['.npy'][data_type][folder_type]]) \
                   <= set(iter_dir(os.path.join(path, 'npy'), '.npy', False)) or reformat_flag:
                print('formatting the data: ', data_type, folder_type)
                self.format(folder_type)
            else:
                print(data_type, folder_type, 'has been processed')

    def download(self, period, include_devices=False):
        categories = []
        for data_type in self.categories:
            categories.extend(config[data_type]['type'].copy())
        if include_devices:
            categories.append('device_types')
        date_dict = self.get_dates()
        self.downloader.export(since=date_dict[period]['since'], until=date_dict[period]['until'], reload=True,
                               save_path=os.path.join(self.default_dir, 'previous' if period == 'gap' else period,
                                                      'csv/'),
                               categories=categories)

    def format(self, period):
        loader = Formatting(os.path.join(self.default_dir, period, 'csv'), add_tihm=period == 'previous')
        loader.sleep_data
        dataloader = Dataloader(loader.activity_data,
                                loader.physiological_data,
                                loader.environmental_data,
                                loader.sleep_data,
                                self.num_days_extended, period == 'previous')

        categories = ['labelled', 'unlabelled'] if period == 'previous' else ['unlabelled']
        for data_type in categories:
            save_path = os.path.join(self.default_dir, period, 'npy', data_type)
            save_mkdir(save_path)
            attr = 'get_{}_data'.format(data_type)
            activity_data, physiological_data, environmental_data, sleep_data, p_ids, labels, dates = getattr(dataloader, attr)()
            np.save(os.path.join(save_path, 'activity.npy'.format(data_type)), activity_data.astype(float))
            np.save(os.path.join(save_path, 'physiological.npy'.format(data_type)), physiological_data)
            np.save(os.path.join(save_path, 'environmental.npy'.format(data_type)), environmental_data)
            np.save(os.path.join(save_path, 'sleep.npy'.format(data_type)), sleep_data)
            np.save(os.path.join(save_path, 'patient_id.npy'), p_ids.astype(str))
            if data_type == 'labelled':
                np.save(os.path.join(save_path, 'label.npy'), labels)
            np.save(os.path.join(save_path, 'dates.npy'), dates)

    def refresh(self, refresh_period=None):
        if refresh_period is None:
            refresh_period = ['current']
        try:
            date_dict = self.get_dates()
        except FileNotFoundError:
            print('Dates file does not exist, start to initialise')
            self.initialise()
            return
        if date_dict['current']['until'] == DT.date.today() - DT.timedelta(days=1):
            print('Data is up-to-date')
            return
        dates_save(refresh=False)
        date_dict = self.get_dates()
        try:
            if date_dict['gap']['until'] > date_dict['gap']['since']:
                self.download('gap')
            self.download('current')
        except TypeError:
            date_backup(True)
            return False
        self.collate()
        for folder in refresh_period:
            self.format(folder)
        date_backup(False)
        return

    def collate(self):
        date_dict = self.get_dates()
        for filename in iter_dir(self.previous_csv_data, split=False):
            if filename not in ['device_types.csv', 'homes.csv', 'patients.csv']:
                previous_data = pd.read_csv(os.path.join(self.previous_csv_data, filename), index_col=0)
                current_data = pd.read_csv(os.path.join(self.current_csv_data, filename), index_col=0)
                current_data = current_data[current_data.start_date != 'start_date']
                previous_data = previous_data[previous_data.start_date != 'start_date']

                current_data.start_date = pd.to_datetime(current_data.start_date)
                current_mask = current_data.start_date.dt.date < date_dict['gap']['until']
                previous_data = pd.concat([previous_data, current_data[current_mask]])
                current_data = current_data[~current_mask]

                current_data.drop_duplicates().to_csv(os.path.join(self.current_csv_data, filename), index=False)
                previous_data.drop_duplicates().to_csv(os.path.join(self.previous_csv_data, filename), index=False)
        return

    @staticmethod
    def get_dates():
        '''
        This function returns the current dates saved in the configurations folder.
        This is an internal function.

        Returns
        ---------

        - dates: dict:
            This dictionary holds the state ('gap', 'current', etc) and the dates.

        '''
        with open(dates_path) as json_file:
            date_dict = json.load(json_file)
        for state in date_dict:
            for time in date_dict[state]:
                date_dict[state][time] = pd.to_datetime(date_dict[state][time])
        return date_dict

    @staticmethod
    def clean_df(path):
        '''
        Use to clean dataframe contains unnamed columns.
        Returns
        -------

        '''
        for filename in iter_dir(path, split=False):
            df = pd.read_csv(os.path.join(path, filename), index_col=0)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.to_csv(os.path.join(path, filename), index=False)