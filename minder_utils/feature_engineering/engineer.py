import pandas as pd
import os
from minder_utils.feature_engineering.configuration import feature_config
from minder_utils.util.decorators import load_save


class Feature_engineer:
    '''
    This class will take the path of raw data as input, and output the following features:
    1. needing to pee more often than usual during the night (nocturia)
    - Compare the frequency of bathroom (Night)

    2. needing to pee suddenly or more urgently than usual
        - Compare the time difference between the triggered last sensor and bathroom sensor

    3. needing to pee more often than usual
        - Compare the frequency of bathroom (Daytime)

    4. a high temperature, or feeling hot and shivery
        - Body temperature (High)

    5. a very low temperature below 36C
        - Body temperature (Low)
    '''

    def __init__(self, raw_data_path='./data/raw_data/'):
        self.raw_data_path = raw_data_path

    def _get_bathroom_activity(self, time_range):
        data = pd.read_csv(os.path.join(self.raw_data_path, 'raw_activity_pir.csv'))[['patient_id',
                                                                                      'start_date', 'location_name']]
        data = data[data.location_name == 'bathroom1']
        data.start_date = pd.to_datetime(data.start_date)
        data = data.set_index('start_date').between_time(*time_range).reset_index()
        data.start_date = data.start_date.dt.date
        data['value'] = 1
        data = data.groupby(['patient_id', 'start_date'])['value'].sum().reset_index()
        data.columns = ['id', 'time', 'value']
        data.time = pd.to_datetime(data.time)
        data['week'] = self.compute_week_number(data.time)
        return data

    def _get_body_temperature(self):
        data = pd.read_csv(os.path.join(self.raw_data_path, 'raw_body_temperature.csv'))[['patient_id',
                                                                                          'start_date', 'value']]
        data.start_date = pd.to_datetime(data.start_date).dt.date
        data = data.groupby(['patient_id', 'start_date'])['value'].mean().reset_index()
        data.columns = ['id', 'time', 'value']
        data.time = pd.to_datetime(data.time)
        data['week'] = self.compute_week_number(data.time)
        return data

    @property
    @load_save(**feature_config.bathroom_night)
    def bathroom_night(self):
        return self._get_bathroom_activity(feature_config.nocturia['time_range'])

    @property
    @load_save(**feature_config.bathroom_daytime)
    def bathroom_daytime(self):
        return self._get_bathroom_activity(feature_config.nocturia['time_range'][::-1])

    @property
    @load_save(**feature_config.body_temperature)
    def body_temperature(self):
        return self._get_body_temperature()

    @staticmethod
    def compute_week_number(df):
        return df.dt.isocalendar().week + (df.dt.isocalendar().year - 2000) * 100

