import pandas as pd
import os
import time
from minder_utils.configurations import config
from .format_util import iter_dir
from minder_utils.download.download import Downloader
import numpy as np


class Formatting:
    """
    Process the data to the following dataframe:

    Patient id, device type, time, value
    """

    def __init__(self, path='./data/raw_data/'):
        self.path = path

        categories_check = ['device_types', 'homes', 'patients']
        if not np.all([os.path.exists(os.path.join(path, category + '.csv')) for category in categories_check]):
            print('Downloading required files for formatting')
            dl = Downloader()
            dl.export(categories=['device_types', 'homes', 'patients'],
                      reload=True, since = None, until = None, save_path=path)
            print('Required files downloaded')

        self.device_type = \
        pd.read_csv(os.path.join(self.path, 'device_types.csv'))[['id', 'type']].set_index('id').to_dict()['type']
        self.config = config
        self.physiological_data = pd.DataFrame(columns=self.config['physiological']['columns'])
        self.activity_data = pd.DataFrame(columns=self.config['activity']['columns'])
        self.environmental_data = pd.DataFrame(columns=self.config['environmental']['columns'])
        self.process_data()

    def process_data(self):
        for name in iter_dir(self.path):
            start_time = time.time()
            print('Processing: {}'.format(name).ljust(50, ' '), end='')
            if name in self.config['physiological']['type']:
                self.process_physiological_data(name)
            elif name in self.config['activity']['type']:
                self.process_activity_data(name)
            elif name in self.config['environmental']['type']:
                self.process_environmental_data(name)
            elif name in self.config['individuals']['text'] or name in self.config['individuals']['measure']:
                print('TODO')
                continue
            end_time = time.time()
            print('Finished in {:.2f} seconds'.format(end_time - start_time))

    def process_physiological_data(self, name):
        """
        process the physiological data, the data will be append to the self.physiological_data

        NOTE:
            the data will be averaged by date and patient id

        :param name: file name to load the data.
            data: the data must contains ['patient_id', 'start_date', 'device_type', 'value', 'unit']
        :return: append to the self.physiological_data
        """
        col_filter = ['patient_id', 'start_date', 'device_type', 'value']
        data = pd.read_csv(os.path.join(self.path, name + '.csv'))
        data.loc[:, 'device_type'] = data.device_type.map(self.device_type)
        try:
            data = getattr(self, 'process_' + name)(data)[col_filter]
        except AttributeError:
            data.device_type += '->' + name[4:]
        data.start_date = pd.to_datetime(data.start_date).dt.date
        data = data[col_filter]
        data = data.groupby(['patient_id', 'start_date', 'device_type']).mean().reset_index()
        data.columns = self.config['physiological']['columns']
        data.location = data.location.apply(lambda x: x.split('->')[-1])
        self.physiological_data = self.physiological_data.append(data)

    def process_activity_data(self, name):
        """
        process the activity data, the data will be append to the self.activity_data

        NOTE:
            Door        -> the values will be set to one (either open or close)
            application -> the values will be set to 1, the location_name will be set based on the
                            the names of the values, e.g. iron-use -> location_name: iron, value: 1
        :param name: the file name to load the data
            the data must contains ['patient_id', 'start_date', 'location_name', 'value']
        :return: append to the self.activity_data
        """
        col_filter = ['patient_id', 'start_date', 'location_name', 'value']
        data = pd.read_csv(os.path.join(self.path, name + '.csv'))
        data = data[data['location_name'] != 'location_name']
        data = getattr(self, 'process_' + name)(data)[col_filter]
        data.columns = self.config['activity']['columns']
        self.activity_data = self.activity_data.append(data)

    def process_environmental_data(self, name):
        """
        process the environmental data, the data will be append to the self.environmental_data

        NOTE:
            the data will be averaged by date and patient id
        :param name: file name to load the data.
            data: the data must contains ['patient_id', 'start_date', 'location_name', 'device_type', 'value', 'unit']
        :return: append to the self.environmental_data
        """
        col_filter = ['patient_id', 'start_date', 'location_name', 'value']
        data = pd.read_csv(os.path.join(self.path, name + '.csv'))
        data = data[data.start_date != 'start_date']
        data.start_date = pd.to_datetime(data.start_date).dt.date
        data = data[col_filter]
        # data.loc[:, 'device_type'] = data.device_type.map(self.device_type)
        data.value = data.value.astype(float)
        data = data.groupby(['patient_id', 'start_date', 'location_name']).mean().reset_index()
        data.columns = self.config['environmental']['columns']
        self.environmental_data = self.environmental_data.append(data)

    def process_raw_door_sensor(self, data):
        data['value'] = 1
        return data

    def process_raw_activity_pir(self, data):
        data['value'] = 1
        return data

    def process_raw_appliance_use(self, data):
        data.location_name = data.value.apply(lambda x: x.split('-')[0])
        data.value = 1
        return data

    def process_raw_blood_pressure(self, data):
        col_filter = ['patient_id', 'start_date', 'device_type', 'value', 'unit']
        blood_pressure = []
        for value in ['systolic_value', 'diastolic_value']:
            tmp_filter = col_filter.copy()
            tmp_filter[3] = value
            tmp_data = data[tmp_filter]
            tmp_data.device_type += '->' + value.split('_')[0]
            tmp_data['value'] = tmp_data[value]
            blood_pressure.append(tmp_data[col_filter])
        return pd.concat(blood_pressure)

    def process_observation_notes(self):
        """
        TODO
            need to be process by NLP maybe
        :return:
        """
        return False

    def process_procedure(self):
        """
        TODO
            need to be process by NLP maybe
        :return:
        """
        return False

    def process_raw_behavioural(self):
        """
        TODO
            need to be process by NLP maybe
        :return:
        """
        return False

    def process_issue(self):
        """
        TODO
        :return:
        """
        return False

    def process_encounter(self):
        """
        TODO
        :return:
        """
        return False

    def process_homes(self):
        """
        TODO
        Data:
        homd id - patient id
        :return:
        """
        return False



if __name__ == '__main__':
    loader = Formatting('./raw_data/')
