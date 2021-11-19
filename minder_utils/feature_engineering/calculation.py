from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
from typing import Union
from scipy.stats import entropy as cal_entropy
from minder_utils.models.outlier_detection import ZScore
from sklearn.preprocessing import StandardScaler
from .util import frequencies_tp
from sklearn.ensemble import IsolationForest

def weekly_compare(df: pd.DataFrame, func, num_previous_week=1) -> dict:
    '''
    Function to compare the values of each patient in current week to previous weeks
    Args:
        df: Dataframe, it should contains at least three columns, which are ['id', 'week', 'value'],
            where the id is the patient ids, week is the numeric numbers got from dt.week, value is the
            sensor readings.
        func: function, used to compare the difference between current week and previous week.
        num_previous_week: int, optional, default is 1, number of previous weeks

    Returns:
        results: dictionary, key is the patient id, value is a list containing the values calculated by func.

    '''
    assert num_previous_week >= 1, 'num_previous_week must be equal or greater than 1'
    num_weeks = df.week.sort_values().unique()
    results = {}
    for p_id in df.id.unique():
        results[p_id] = []
    for idx, week in enumerate(num_weeks):
        if idx < num_previous_week:
            continue
        current_week = df[df.week == week]
        previous_week = df[df.week.isin([week - i for i in range(1, num_previous_week + 1)])]
        for p_id in df.id.unique():
            previous_patient_data = previous_week[previous_week.id == p_id].value.to_numpy()
            current_patient_data = current_week[current_week.id == p_id].value.to_numpy()
            if current_patient_data.shape[0] == 0 or previous_patient_data.shape[0] == 0:
                continue
            try:
                results[p_id].append(func(current_patient_data, previous_patient_data))
            except ValueError:
                pass
    return results


def threshold_compare(df: pd.DataFrame, func='>', threshold=36) -> pd.DataFrame:
    '''
    Function to filter the dataframe by threshold
    Args:
        df:
        func:
        threshold:

    Returns:

    '''
    if func == '>':
        return df[df.value > threshold]
    elif func == '<':
        return df[df.value < threshold]


def calculate_entropy(df: pd.DataFrame, sensors: Union[list, str]) -> pd.DataFrame:
    '''
    Return a dataframe with research id, week id, and entropy value
    based on list of sensors given. If resulting activity count of
    any of the sensors given in the list is zero, the value of the
    entropy will be NaN.

    Args:
        df: Dataframe, contains at least 4 columns ['id', 'week', 'location', 'value']
        sensors: List or string, if list,  will calculate the entropy based on the list
            of sensors; if string, only accept 'all', which means use all sensors.

    Returns:

    '''
    assert len(sensors) >= 2, 'need at least two sensors to calculate the entropy'

    # Filter the sensors
    if isinstance(sensors, list):
        df = df[df.location.isin(sensors)]
    elif isinstance(sensors, str):
        assert sensors == 'all', 'only accept all as a string input for sensors'

    # Sum the the number of readings of sensors weekly
    sensor_summation = df.groupby(['id', 'week'])['value'].sum().reset_index()
    sensor_summation.columns = ['id', 'week', 'summation']

    # Merge with existing dataframe
    df = pd.merge(df, sensor_summation)

    # Calculate the probabilities
    df['probabilities'] = df['value'] / df['summation']

    # entropy function used in groupby
    def cal_entropy_groupby(x):
        x = cal_entropy(list(x))
        return x
    
    # Calculate the entropy
    df = df.groupby(by=['id', 'week'])['probabilities'].apply(cal_entropy_groupby).reset_index()

    df.columns = ['id', 'week', 'value']
    df['location'] = 'entropy'
    return df


def kolmogorov_smirnov(freq1, freq2):
    return ks_2samp(freq1, freq2)





def anomaly_detection_freq(input_df, outlier_class, tp_for_outlier_hours = 3, baseline_length_days = 7,
                           baseline_offset_days = 0 ):
    '''
    Given an outlier function, and an input, this function calculates an outlier score
    for every point based on a window of ```baseline_length_days``` days. Because this
    function fits the class for every new point, using a complicated outlier detection
    class is not possible. Please consider using a light class.
    
    Arguments
    ---------

    - input_df: pandas dataframe:
        This dataframe must have the columns ```'time'``` and ```'location'```. This is the
        data to calculate outlier scores on.

    - outlier_class: class or string
        This is the class that will be used to calculate the outlier scores. This class must have
        the functions ```.fit()``` to fit the class and ```.decision_function()``` to produce the
        outlier scores. Inputs to these functions will always be 2d. The input to ```.fit()``` will
        be an array of shape ```(N_t, N_f)``` where ```N_t``` is the number of points that fit in the 
        ```baseline_length_days```. Each point will represent the frequencies of location visits for 
        a given ```tp_for_outlier_hours``` hour time period. The input to ```.decision_function()```
        will be an array of shape ```(1, N_f)``` as it will be a single point.
        If string, make sure it is one of ['zscore', 'isolation_forest']

    - tp_for_outlier_hours: int:
        This is the number of hours to aggregate the frequency data by. This is the ```tp``` 
        input to the function ```minder_utils.feature_engineering.util.frequencies_tp```.

    - baseline_length_days: integer:
        This is the length of the baseline in days that will be used. This value is used when finding
        the ```baseline_length_days``` complete days of the frequency data to use as a baseline.
    
    - baseline_offset_days: integer:
        This is the offset to the baseline period. ```0``` corresponds to a time period ending the morning of the
        current date being calculated on.


    Returns
    ---------
    '''

    
    frequency_df, locations = frequencies_tp(input_df, tp = tp_for_outlier_hours, return_locations=True)
    X = frequency_df[locations].values
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    out = np.zeros(frequency_df.shape[0])

    dates = frequency_df['time'].values

    baseline_length_tps = int(np.ceil(24/tp_for_outlier_hours * baseline_length_days))
    baseline_offset_tps = int(np.ceil(24/tp_for_outlier_hours * baseline_offset_days))

    if outlier_class == 'zscore':
        outlier_class = ZScore()
    elif outlier_class == 'isolation_forest':
        outlier_class = IsolationForest()


    for nd, date in enumerate(dates):
        index_baseline_end = np.where(dates <= date)[0][-1]
        
        index_baseline_end = index_baseline_end - baseline_offset_tps
        index_baseline_start = index_baseline_end - baseline_length_tps

        if index_baseline_start < 0:
            out[nd] = np.NAN

        else:
            X_s_input = X_s[index_baseline_start:index_baseline_end]
            X_s_current = X_s[nd].reshape(1, -1)

            outlier_class.fit(X_s_input)
            outlier_scores = outlier_class.decision_function(X_s_current)
            out[nd] = outlier_scores


    frequency_df['outlier_score'] = out

    return frequency_df