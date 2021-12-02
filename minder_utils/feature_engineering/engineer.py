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

    def __init__(self, formatter, agg_method = 'sum'):
        self.formatter = formatter
        self.agg_method = agg_method

    @property
    def info(self):
        return {
            'bathroom_night': 'Bathroom activity during the night',
            'bathroom_daytime': 'Bathroom activity during the day',
             # 'bathroom_urgent': 'TODO',
            'body_temperature': 'Mean of body temperature of the participant during the week',
            'entropy': 'Entropy of the activity',
            # 'raw_activity': 'Raw activity data (weekly)',
            'entropy_rate': 'Entropy rate of markov chain over the week',
            'bathroom_night_ma': 'Moving average of bathroom activity during the night',
            'bathroom_night_ma_delta': 'Delta in the moving average of bathroom activity during the night',
            'bathroom_daytime_ma': 'Moving average of bathroom activity during the day',
            'bathroom_daytime_ma_delta': 'Delta in the moving average of bathroom activity during the day',
            'bathroom_urgent_reverse_percentage': 'Reverse percentile of the time to the bathroom',
            'outlier_score_activity': 'Outlier score of the activity',
            'rp_location_time_delta': 'Reverse percentile of the time between activities',
        }

    @property
    def agg_info(self):
        '''
        These are the time aggregations of each of the datasets.
        '''
        return {'evently': ['rp_location_time_delta'], 

                'daily': ['bathroom_night',
                          'bathroom_daytime',
                          'outlier_score_activity',
                          'bathroom_night_ma',
                          'bathroom_night_ma_delta',
                          'bathroom_daytime_ma',
                         #'bathroom_urgent',
                          'bathroom_urgent_reverse_percentage',
                          'bathroom_daytime_ma_delta'], 

                'weekly': ['body_temperature', 
                           'entropy', 
                           'entropy_rate']}



    

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
        data = get_bathroom_delta(self.formatter.activity_data, rp_single_location_delta, 'bathroom_urgent_reverse_percentage')
        def value_group_by(x):
            x[np.where(x == -1)] = np.nan
            x = np.nanmean(x)
            return x
        data['value'] = data['value'].apply(value_group_by)
        return data

    @property
    @load_save(**feature_config['body_temperature']['save'])
    def body_temperature(self):
        return get_body_temperature(self.formatter.physiological_data)

    @property
    @load_save(**feature_config['entropy']['save'])
    def entropy(self):
        return calculate_entropy(self.formatter.activity_data, feature_config['entropy']['sensors'])

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
        if feature_config['activity']['features'] is None:
            features = self.info.keys()
        else:
            features = feature_config['activity']['features']
        for feat in features:
            data.append(getattr(self, feat)[['id', 'week', 'location', 'value']])
        data = pd.concat(data)
        if self.agg_method == 'sum':
            data = data.groupby(['id', 'week', 'location'])['value'].sum().reset_index()
        elif self.agg_method == 'median':
            data = data.groupby(['id', 'week', 'location'])['value'].median().reset_index()
        elif self.agg_method == 'mean':
            data = data.groupby(['id', 'week', 'location'])['value'].mean().reset_index()
        else:
            raise TypeError('agg_method={} is not implemented'.format(self.agg_method))
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

    def activity_specific_agg(self, agg='daily', load_smaller_aggs = False):
        
        accepted_agg = ['evently', 'daily', 'weekly']
        if not agg in accepted_agg:
            raise TypeError('Please use an agg from the list {}'.format(accepted_agg))

        data = []

        '''
        if feature_config['activity_{}'.format(agg)]['features'] is None:
            features = self.info.keys()
        
        else:
            features = feature_config['activity_{}'.format(agg)]['features']
        '''

        features = self.agg_info[agg]
        if load_smaller_aggs:
            if agg == 'weekly':
                features.extend(self.agg_info['daily'])
                #features.extend(self.agg_info['evently'])
            elif agg == 'daily':
                features.extend(self.agg_info['evently'])
        for feat in features:
            if agg =='weekly':
                feat_data = getattr(self, feat)[['id', 'week', 'location', 'value']]
            elif agg =='daily':
                feat_data = getattr(self, feat)[['id', 'week', 'time', 'location', 'value']]
            elif agg =='evently':
                feat_data = getattr(self, feat)
            data.append(feat_data)
        data = pd.concat(data)
        if agg == 'weekly':
            data['time'] = week_to_date(data['week'])
        data['time'] = pd.to_datetime(data['time'])
        if not agg == 'evently':
            columns_agg = ['id', 'week', 'location']
            grouper = pd.Grouper(key = 'time', freq = '1d' if agg=='daily' else '1W', dropna = False)
            columns_agg.append(grouper)
            if self.agg_method == 'sum':
                data = data.groupby(columns_agg)['value'].sum().reset_index()
            elif self.agg_method == 'median':
                data = data.groupby(columns_agg)['value'].median().reset_index()
            elif self.agg_method == 'mean':
                data = data.groupby(columns_agg)['value'].mean().reset_index()
            else:
                raise TypeError('agg_method={} is not implemented'.format(self.agg_method))

        return data

    @property
    @load_save(**feature_config['activity_daily']['save'])
    def activity_daily(self):
        return self.activity_specific_agg(agg='daily')

    @property
    @load_save(**feature_config['activity_evently']['save'])
    def activity_evently(self):
        return self.activity_specific_agg(agg='evently')

    @property
    @load_save(**feature_config['activity_weekly']['save'])
    def activity_weekly(self):
        return self.activity_specific_agg(agg='weekly')