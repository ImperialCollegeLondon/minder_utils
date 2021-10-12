from formatting.format_util import iter_dir
import os
import datetime as DT
from download.download import Downloader
from formatting.formatting import Formatting
from dataloader.dataloader import Dataloader
from formatting.standardisation import standardise_activity_data
import numpy as np
from util import save_mkdir


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
    def __init__(self, data_type='activity', num_days_extended=3):
        '''

        @param data_type: activity, #TODO environmental, physiological
        @param num_days_extended: for uti only, how many consecutive days to be labelled
        '''
        self.data_type = data_type
        self.num_days_extended = num_days_extended
        self.downloader = Downloader()
        self.default_dir = './data/weekly_test'
        save_mkdir(self.default_dir)

    @property
    def previous_data(self):
        return os.path.join(self.default_dir, 'previous', 'npy')

    @property
    def weekly_data(self):
        return os.path.join(self.default_dir, 'weekly', 'npy')

    def check_exist(self, path):
        print('Check existing data ...')
        check_list = {
            '.csv': {'activity': ['raw_door_sensor', 'raw_appliance_use', 'raw_activity_pir', 'device_types']},
            '.npy': {'activity': {'weekly': ['unlabelled', 'patient_id', 'dates'],
                                  'previous': ['unlabelled', 'patient_id', 'dates', 'X', 'y']}},
        }
        folder_type = 'weekly' if 'week' in path.split('/')[-1] else 'previous'
        save_mkdir(os.path.join(path, 'csv'))
        save_mkdir(os.path.join(path, 'npy'))
        reformat_flag = False
        for data_type in ['activity']:
            # Check the csv file
            if not set([ele + '.csv' for ele in check_list['.csv'][data_type]])\
                   <= set(iter_dir(os.path.join(path, 'csv'), '.csv', False)):
                print(data_type, folder_type, 'raw data does not exist, start to download')
                self.download(folder_type, data_type)
                reformat_flag = True
            else:
                print(data_type, folder_type, 'is already downloaded')

            # Check the npy file
            if not set([ele + '.npy' for ele in check_list['.npy'][data_type][folder_type]])\
                   <= set(iter_dir(os.path.join(path, 'npy'), '.npy', False)) or reformat_flag:
                print('formatting the data: ', data_type, folder_type)
                self.format(folder_type)
            else:
                print(data_type, folder_type, 'has been processed')

    def download(self, period, data_type):
        categories = {
            'activity': ['raw_door_sensor', 'raw_appliance_use', 'raw_activity_pir']
        }
        since = DT.date.today() - DT.timedelta(days=7) if period == 'weekly' else None
        self.downloader.export(since=since, reload=True, save_path=os.path.join(self.default_dir, period, 'csv/'),
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

    def load_data(self, reload_weekly=False, reload_all=False):
        if reload_weekly:
            self.download('weekly', 'activity')
            self.format('weekly')
        if reload_all:
            self.download('previous', 'activity')
            self.format('previous')
        for folder in ['weekly', 'previous']:
            path = os.path.join(self.default_dir, folder)
            self.check_exist(path)

