import pandas as pd
import numpy as np
from minder_utils.configurations import feature_config, config
from minder_utils.util.decorators import load_save
from minder_utils.feature_engineering.compare_functions import *
from minder_utils.feature_engineering.TimeFunctions import single_location_delta


class Feature_engineer:
    '''
    Take the formatting as input, re-construct the data in weekly format

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

    def __init__(self, formatter):
        self.formatter = formatter

    def _get_bathroom_activity(self, time_range, name):
        data = self.formatter.activity_data
        data = data[data.location == 'bathroom1'][['id', 'time', 'value']]
        data.time = pd.to_datetime(data.time)
        data = data.set_index('time').between_time(*time_range).reset_index()
        data.time = data.time.dt.date
        data = data.groupby(['id', 'time'])['value'].sum().reset_index()
        data['week'] = self.compute_week_number(data.time)
        data['location'] = name
        return data

    def _get_body_temperature(self):
        data = self.formatter.physiological_data
        data = data[data.location == 'body_temperature'][['id', 'time', 'value']]
        data.time = pd.to_datetime(data.time).dt.date
        data = data.groupby(['id', 'time'])['value'].mean().reset_index()
        data['week'] = self.compute_week_number(data.time)
        data['location'] = 'body_temperature'
        return data

    def _get_weekly_activity_data(self):
        data = self.formatter.activity_data
        data.time = pd.to_datetime(data.time).dt.date
        data = data.groupby(['id', 'time', 'location'])['value'].sum().reset_index()
        data['week'] = self.compute_week_number(data.time)
        data = data[data.location.isin(config['activity']['sensors'])]
        return data

    def _get_bathroom_delta(self):
        data = self.formatter.activity_data
        data.time = pd.to_datetime(data.time).dt.date
        results = {}
        for p_id in data.id.unique():
            p_data = single_location_delta(data[data.id == p_id], 'bathroom1',
                                           recall_value=feature_config['bathroom_urgent']['recall_value'])
            if len(p_data) > 0:
                results[p_id] = p_data
        results = pd.DataFrame([(i, j, results[i][j].astype(float)) for i in results for j in results[i]],
                               columns=['id', 'time', 'value'])
        results['location'] = 'bathroom_urgent'
        return results

    @property
    @load_save(**feature_config['bathroom_night']['save'])
    def bathroom_night(self):
        return self._get_bathroom_activity(feature_config['nocturia']['time_range'], 'bathroom_night')

    @property
    @load_save(**feature_config['bathroom_daytime']['save'])
    def bathroom_daytime(self):
        return self._get_bathroom_activity(feature_config['nocturia']['time_range'][::-1], 'bathroom_daytime')

    @property
    @load_save(**feature_config['bathroom_urgent']['save'])
    def bathroom_urgent(self):
        return self._get_bathroom_delta()

    @property
    @load_save(**feature_config['body_temperature']['save'])
    def body_temperature(self):
        return self._get_body_temperature()

    @property
    @load_save(**feature_config['raw_activity']['save'])
    def raw_activity(self):
        return self._get_weekly_activity_data()

    @property
    @load_save(**feature_config['activity']['save'])
    def activity(self):
        data = []
        for feat in feature_config['activity']['features']:
            data.append(getattr(self, feat))
        data = pd.concat(data)
        data = data.groupby(['id', 'week', 'location'])['value'].sum().reset_index()
        return data.pivot_table(index=['id', 'week'], columns='location',
                                values='value').reset_index().replace(np.nan, 0)

    @staticmethod
    def compute_week_number(df):
        df = pd.to_datetime(df)
        return df.dt.isocalendar().week + (df.dt.isocalendar().year - 2000) * 100
