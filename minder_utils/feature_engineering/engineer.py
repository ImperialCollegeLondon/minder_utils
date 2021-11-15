import pandas as pd
import os
from minder_utils.configurations import feature_config
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

    def __init__(self, formater):
        self.formater = formater

    def _get_bathroom_activity(self, time_range):
        data = self.formater.activity_data
        data = data[data.location == 'bathroom1'][['id', 'time', 'value']]
        data.time = pd.to_datetime(data.time)
        data = data.set_index('time').between_time(*time_range).reset_index()
        data.time = data.time.dt.date
        data = data.groupby(['id', 'time'])['value'].sum().reset_index()
        data['week'] = self.compute_week_number(data.time)
        return data

    def _get_body_temperature(self):
        data = self.formater.physiological_data
        data = data[data.location == 'body_temperature'][['id', 'time', 'value']]
        data.time = pd.to_datetime(data.time).dt.date
        data = data.groupby(['id', 'time'])['value'].mean().reset_index()
        data['week'] = self.compute_week_number(data.time)
        return data

    @property
    @load_save(**feature_config['bathroom_night']['save'])
    def bathroom_night(self):
        return self._get_bathroom_activity(feature_config['nocturia']['time_range'])

    @property
    @load_save(**feature_config['bathroom_daytime']['save'])
    def bathroom_daytime(self):
        return self._get_bathroom_activity(feature_config['nocturia']['time_range'][::-1])

    @property
    @load_save(**feature_config['body_temperature']['save'])
    def body_temperature(self):
        return self._get_body_temperature()

    @staticmethod
    def compute_week_number(df):
        df = pd.to_datetime(df)
        return df.dt.isocalendar().week + (df.dt.isocalendar().year - 2000) * 100

