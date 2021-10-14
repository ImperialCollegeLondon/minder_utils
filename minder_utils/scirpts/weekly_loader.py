from ..formatting.format_util import iter_dir
import os
import datetime as DT
from ..download.download import Downloader
from ..formatting.formatting import Formatting
from ..dataloader.dataloader import Dataloader
from ..formatting.standardisation import standardise_activity_data
import numpy as np
from ..util import save_mkdir, delete_dir
import json
from ..settings import dates_save
import pandas as pd
from minder_utils.configurations import dates_path


class Weekly_dataloader:
    """
    Support UTI only
    This class will
        - download all previous data if it have not been downloaded before
            - will be saved as labelled data and unlabelled data
        - download the latest weekly data
        - reformat all the data into following (N is the number of samples)
            - activity data: N * 3 * 8 * 20
            - environmental data: #TODO
            - physiological data: #TODO
    """

    def __init__(self, data_type='activity', save_dir=os.path.join('data', 'weekly_test'), num_days_extended=3):
        '''

        @param data_type: activity, #TODO environmental, physiological
        @param num_days_extended: for uti only, how many consecutive days to be labelled
        '''
        self.data_type = data_type
        self.num_days_extended = num_days_extended
        self.downloader = Downloader()
        self.default_dir = save_dir
        save_mkdir(self.default_dir)

    @property
    def previous_data(self):
        return os.path.join(self.default_dir, 'previous', 'npy')

    @property
    def current_data(self):
        return os.path.join(self.default_dir, 'current', 'npy')

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
            delete_dir(os.path.join(self.default_dir, folder, 'npy'))
            save_mkdir(os.path.join(self.default_dir, folder, 'csv'))
            save_mkdir(os.path.join(self.default_dir, folder, 'npy'))
            self.download(folder, 'activity', include_devices=True)
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

    def download(self, period, data_type='activity', include_devices=False):
        categories = {
            'activity': ['raw_door_sensor', 'raw_appliance_use', 'raw_activity_pir']
        }
        if include_devices:
            categories[data_type].append('device_types')
        date_dict = self.get_dates()
        self.downloader.export(since=date_dict[period]['since'], until=date_dict[period]['until'], reload=True,
                               save_path=os.path.join(self.default_dir, 'previous' if period == 'gap' else period,
                                                      'csv/'),
                               categories=categories[data_type])

    def format(self, period):
        loader = Formatting(os.path.join(self.default_dir, period, 'csv'))
        dataloader = Dataloader(standardise_activity_data(loader.activity_data),
                                self.num_days_extended, period == 'previous')
        save_path = os.path.join(self.default_dir, period, 'npy')

        data, p_ids, dates = dataloader.get_unlabelled_data()
        np.save(os.path.join(save_path, 'unlabelled.npy'), data)
        np.save(os.path.join(save_path, 'patient_id.npy'), p_ids)
        np.save(os.path.join(save_path, 'dates.npy'), dates)
        if period == 'previous':
            x, y, z = [], [], []
            for i, j, k in dataloader.iterate_data():
                x.append(i)
                y.append(j)
                z.append(k)
            np.save(os.path.join(save_path, 'X.npy'), np.concatenate(x))
            np.save(os.path.join(save_path, 'y.npy'), np.concatenate(y))
            np.save(os.path.join(save_path, 'label_ids.npy'), np.concatenate(z))

    def refresh(self):
        date_dict = self.get_dates()
        if date_dict['current']['until'] == DT.date.today() - DT.timedelta(days=1):
            print('Data is up-to-date')
            return
        dates_save(refresh=False)
        date_dict = self.get_dates()
        if date_dict['gap']['until'] > date_dict['gap']['since']:
            self.download('gap')
        self.download('current')
        self.collate()
        return

    def collate(self):
        date_dict = self.get_dates()
        for filename in iter_dir(self.previous_csv_data, split=False):
            if filename not in ['device_types.csv']:
                previous_data = pd.read_csv(os.path.join(self.previous_csv_data, filename))
                current_data = pd.read_csv(os.path.join(self.current_csv_data, filename))

                current_data.start_date = pd.to_datetime(current_data.start_date)
                current_mask = current_data.start_date.dt.date < date_dict['gap']['until']
                previous_data = pd.concat([previous_data, current_data[current_mask]])
                current_data = current_data[~current_mask]

                current_data.drop_duplicates().to_csv(os.path.join(self.current_csv_data, filename))
                previous_data.drop_duplicates().to_csv(os.path.join(self.previous_csv_data, filename))

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
