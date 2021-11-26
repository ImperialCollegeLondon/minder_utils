import numpy as np
from minder_utils.util.decorators import load_save
from .adding_features import *
from .calculation import calculate_entropy, anomaly_detection_freq
from .TimeFunctions import single_location_delta, rp_single_location_delta
from .util import week_to_date


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

    @property
    @load_save(**feature_config['bathroom_night']['save'])
    def bathroom_night(self):
        return get_bathroom_activity(self.formatter.activity_data, feature_config['nocturia']['time_range'], 'bathroom_night')

    @property
    @load_save(**feature_config['bathroom_night_ma']['save'])
    def bathroom_night_ma(self):

        def get_moving_average_groupby(x):
            x = get_moving_average(x, 
                                  w=feature_config['bathroom_night_ma']['w'], 
                                  name='bathroom_night_ma')
            return x

        return (self.bathroom_night).groupby('id').apply(get_moving_average_groupby)


    @property
    @load_save(**feature_config['bathroom_night_ma_delta']['save'])
    def bathroom_night_ma_delta(self):

        def get_value_delta_groupby(x):
            x = get_value_delta(x,
                                name='bathroom_night_ma_delta')
            return x

        return (self.bathroom_night_ma).groupby('id').apply(get_value_delta_groupby)

    @property
    @load_save(**feature_config['bathroom_daytime']['save'])
    def bathroom_daytime(self):
        return get_bathroom_activity(self.formatter.activity_data, feature_config['nocturia']['time_range'][::-1], 'bathroom_daytime')

    @property
    @load_save(**feature_config['bathroom_daytime_ma']['save'])
    def bathroom_daytime_ma(self):

        def get_moving_average_groupby(x):
            x = get_moving_average(x, 
                                  w=feature_config['bathroom_daytime_ma']['w'], 
                                  name='bathroom_daytime_ma')
            return x

        return (self.bathroom_daytime).groupby(by='id').apply(get_moving_average_groupby)



    @property
    @load_save(**feature_config['bathroom_daytime_ma_delta']['save'])
    def bathroom_daytime_ma_delta(self):

        def get_value_delta_groupby(x):
            x = get_value_delta(x,
                                name='bathroom_daytime_ma_delta')
            return x

        return (self.bathroom_daytime_ma).groupby('id').apply(get_value_delta_groupby)



    @property
    @load_save(**feature_config['bathroom_urgent']['save'])
    def bathroom_urgent(self):
        return get_bathroom_delta(self.formatter.activity_data, single_location_delta, 'bathroom_urgent')

    @property
    @load_save(**feature_config['bathroom_urgent_reverse_percentage']['save'])
    def bathroom_urgent_reverse_percentage(self):
        return get_bathroom_delta(self.formatter.activity_data, rp_single_location_delta, 'bathroom_urgent_reverse_percentage')

    @property
    @load_save(**feature_config['body_temperature']['save'])
    def body_temperature(self):
        return get_body_temperature(self.formatter.physiological_data)

    @property
    @load_save(**feature_config['entropy']['save'])
    def entropy(self):
        return calculate_entropy(self.raw_activity, feature_config['entropy']['sensors'])

    @property
    @load_save(**feature_config['entropy_rate']['save'])
    def entropy_rate(self):
        return get_entropy_rate(df=self.formatter.activity_data, 
                                sensors=feature_config['entropy_rate']['sensors'], 
                                name='entropy_rate')

    @property
    @load_save(**feature_config['raw_activity']['save'])
    def raw_activity(self):
        return get_weekly_activity_data(self.formatter.activity_data)

    @property
    @load_save(**feature_config['activity']['save'])
    def activity(self):
        data = []
        for feat in feature_config['activity']['features']:
            data.append(getattr(self, feat)[['id', 'week', 'location', 'value']])
        data = pd.concat(data)
        data = data.groupby(['id', 'week', 'location'])['value'].sum().reset_index()
        data = data.pivot_table(index=['id', 'week'], columns='location',
                                values='value').reset_index().replace(np.nan, 0)
        data['time'] = week_to_date(data['week'])
        return data

    @property
    @load_save(**feature_config['outlier_score_activity']['save'])
    def outlier_score_activity(self):
        return get_outlier_freq(self.formatter.activity_data, anomaly_detection_freq, 'outlier_score_activity')

    @property
    @load_save(**feature_config['rp_location_time_delta']['save'])
    def rp_location_time_delta(self):
        print('This might take a bit of time...')
        return get_subject_rp_location_delta(data=self.formatter.activity_data, 
                                              columns = {'subject':'id', 'time':'time', 'location':'location'}, 
                                              baseline_length_days = feature_config['rp_location_time_delta']['baseline_length_days'],
                                              baseline_offset_days = feature_config['rp_location_time_delta']['baseline_offset_days'],
                                              all_loc_as_baseline = feature_config['rp_location_time_delta']['all_loc_as_baseline'],
                                              name='rp_location_time_delta')

