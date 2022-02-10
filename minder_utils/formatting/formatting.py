import pandas as pd
import os
import time
from minder_utils.configurations import config
from .format_util import iter_dir
from minder_utils.download.download import Downloader
from minder_utils.util.decorators import load_save
from minder_utils.formatting.format_tihm import format_tihm_data
import numpy as np
from minder_utils.util.util import reformat_path
from .label import label_dataframe


class Formatting:
    """
    Process the data to the following dataframe:

    Patient id, device type, time, value
    """

    def __init__(self, path=os.path.join('./data', 'raw_data'), add_tihm=None):
        self.path = reformat_path(path)
        self.add_tihm = add_tihm
        self.activity_nice_locations = {
                                        'hallway': 'Hallway', 
                                        'kitchen': 'Kitchen', 
                                        'lounge':'Lounge', 
                                        'bathroom1': 'Bathroom', 
                                        'bedroom1':'Bedroom',
                                        'kettle': 'Kettle',
                                        'toaster': 'Toaster',
                                        'fridge door': 'Fridge Door',
                                        'back door': 'Back Door',
                                        'front door': 'Front Door',
                                        'microwave': 'Microwave',
                                        'study': 'Study',
                                        'dining room': 'Dining Room',
                                        'living room': 'Living Room',
                                        'iron': 'Iron',
                                        'corridor1': 'Corridor',
                                        'WC1': 'WC',
                                        'main door': 'Main Door',
                                        'utility': 'Utility', 
                                        'office': 'Office', 
                                        'multi': 'Multi', 
                                        'conservatory': 'Conservatory',
                                        'garage': 'Garage', 
                                        'secondary': 'Secondary', 
                                        'cellar': 'Cellar'
                                        }

        categories_check = ['device_types', 'homes', 'patients']
        if not np.all([os.path.exists(os.path.join(path, category + '.csv')) for category in categories_check]):
            print('Downloading required files for formatting')
            dl = Downloader()
            dl.export(categories=['device_types', 'homes', 'patients'],
                      reload=True, since=None, until=None, save_path=path, append = False)
            print('Required files downloaded')

        self.device_type = \
            pd.read_csv(os.path.join(self.path, 'device_types.csv'))[['id', 'type']].set_index('id').to_dict()['type']
        self.config = config

    @property
    @load_save(**config['physiological']['save'])
    def physiological_data(self):
        add_tihm = config['physiological']['add_tihm'] if self.add_tihm is None else self.add_tihm
        if add_tihm:
            data = self.process_data('physiological')
            tihm_data = format_tihm_data()
            return pd.concat([data, tihm_data['physiological']])
        return label_dataframe(self.process_data('physiological').drop_duplicates())

    @property
    @load_save(**config['activity']['save'])
    def activity_data(self):
        add_tihm = config['activity']['add_tihm'] if self.add_tihm is None else self.add_tihm
        if add_tihm:
            data = self.process_data('activity')
            tihm_data = format_tihm_data()
            return pd.concat([data, tihm_data['activity']]).drop_duplicates().sort_values('time')
        return label_dataframe(self.process_data('activity')).sort_values('time')

    @property
    @load_save(**config['environmental']['save'])
    def environmental_data(self):
        return label_dataframe(self.process_data('environmental'))

    @property
    @load_save(**config['sleep']['save'])
    def sleep_data(self):
        return label_dataframe(self.process_data('sleep')).sort_values('time').reset_index(drop=True)

    def process_data(self, datatype):
        assert datatype in ['physiological', 'activity', 'environmental', 'sleep'], 'not a valid type'
        process_func = getattr(self, 'process_{}_data'.format(datatype))
        dataframe = pd.DataFrame(columns=self.config[datatype]['columns'])
        for name in iter_dir(self.path):
            start_time = time.time()
            print('Processing: {} ------->  {}'.format(datatype, name).ljust(80, ' '), end='')
            if name in self.config[datatype]['type']:
                dataframe = process_func(name, dataframe)
            end_time = time.time()
            print('Finished in {:.2f} seconds'.format(end_time - start_time))
        return dataframe

    def process_sleep_data(self, name, df):
        '''
        This function will process the sleep data.

        '''
        col_filter = ['patient_id', 'start_date']
        categorical_columns = self.config['sleep']['categorical_columns']
        value_columns = self.config['sleep']['value_columns']
        data_adding = pd.read_csv(os.path.join(self.path, name + '.csv'))
        categorical_columns = [column for column in categorical_columns if column in list(data_adding.columns)]
        if len(categorical_columns) != 0:
            data_cat = data_adding[col_filter+categorical_columns].copy()
            data_cat = pd.melt(data_cat.merge(
                                                pd.get_dummies(data_cat[categorical_columns]),
                                                left_index=True, right_index=True
                                                            ).drop(categorical_columns,
                                                                    axis=1),
                                id_vars=col_filter,
                                var_name='location',
                                value_name='value')
            data_cat = data_cat[data_cat.value != 0]
            data_cat = data_cat[data_cat['value'].notna()]
            data_cat.value = data_cat.value.astype(float)
        else:
            data_cat = None

        value_columns = [column for column in value_columns if column in list(data_adding.columns)]
        if len(value_columns) != 0:
            data_val = data_adding[col_filter+value_columns].copy()
            data_val = pd.melt(data_val,
                            id_vars=col_filter,
                            var_name='location',
                            value_name='value')
            data_val = data_val[data_val['value'].notna()]
            data_val.value = data_val.value.astype(float)
        else:
            data_val = None

        if (data_val is None) and (data_cat is None):
            return df

        data_out = pd.concat([data_cat, data_val])

        data_out.columns = self.config['sleep']['columns']
        data_out.time = pd.to_datetime(data_out.time, utc=True)

        return pd.concat([df, data_out])

    def process_physiological_data(self, name, df):
        """
        process the physiological data, the data will be append to the self.physiological_data

        NOTE:
            the data will be averaged by date and patient id

        :param name: string, file name to load the data.
            data: the data must contains ['patient_id', 'start_date', 'device_type', 'value', 'unit']
        :param df: dataframe, dataframe to append the data
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
        data.time = pd.to_datetime(data.time, utc=True)
        return df.append(data)

    def process_activity_data(self, name, df):
        """
        process the activity data, the data will be append to the self.activity_data

        NOTE:
            Door        -> the values will be set to one (either open or close)
            application -> the values will be set to 1, the location_name will be set based on the
                            the names of the values, e.g. iron-use -> location_name: iron, value: 1
        :param name: the file name to load the data
            the data must contains ['patient_id', 'start_date', 'location_name', 'value']
        :param df: dataframe, dataframe to append the data
        :return: append to the self.activity_data
        """
        col_filter = ['patient_id', 'start_date', 'location_name', 'value']
        data = pd.read_csv(os.path.join(self.path, name + '.csv'))
        data = data[data['location_name'] != 'location_name']
        data = getattr(self, 'process_' + name)(data)[col_filter]
        data.columns = self.config['activity']['columns']
        data.time = pd.to_datetime(data.time, utc=True)
        return df.append(data)

    def process_environmental_data(self, name, df):
        """
        process the environmental data, the data will be append to the self.environmental_data

        NOTE:
            the data will be averaged by date and patient id
        :param name: file name to load the data.
            data: the data must contains ['patient_id', 'start_date', 'location_name', 'device_type', 'value', 'unit']
        :param df: dataframe, dataframe to append the data
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
        data.time = pd.to_datetime(data.time, utc=True)
        return df.append(data)

    @staticmethod
    def process_raw_door_sensor(data):
        data['value'] = 1
        return data

    @staticmethod
    def process_raw_activity_pir(data):
        data['value'] = 1
        return data

    @staticmethod
    def process_raw_appliance_use(data):
        data.location_name = data.value.apply(lambda x: x.split('-')[0])
        data.value = 1
        return data

    @staticmethod
    def process_raw_blood_pressure(data):
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
