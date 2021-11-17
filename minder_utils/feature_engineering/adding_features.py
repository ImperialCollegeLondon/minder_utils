from scipy.stats import entropy as cal_entropy
from minder_utils.configurations import feature_config, config
from .util import *
import pandas as pd


def calculate_entropy(df: pd.DataFrame, sensors: list) -> pd.DataFrame:
    '''
    Return a dataframe with research id, week id, and entropy value
    based on list of sensors given. If resulting activity count of
    any of the sensors given in the list is zero, the value of the
    entropy will be NaN.

    Args:
        df: Dataframe, contains at least 4 columns ['id', 'week', 'location', 'value']
        sensors: List, the

    Returns:

    '''
    # Filter the sensors
    df = df[df.location.isin(sensors)]
    # Sum the the number of readings of sensors weekly
    sensor_summation = df.groupby(['id', 'week'])['value'].sum().reset_index()
    sensor_summation.columns = ['id', 'week', 'summation']

    # Merge with existing dataframe
    df = pd.merge(df, sensor_summation)

    # Calculate the probabilities
    df['probabilities'] = df['value'] / df['summation']

    # Calculate the entropy
    df = df.groupby(['id', 'week'])['probabilities'].apply(cal_entropy).reset_index()
    df.columns = ['id', 'week', 'value']
    df['location'] = 'entropy'
    return df


def get_bathroom_activity(data, time_range, name):
    data = data[data.location == 'bathroom1'][['id', 'time', 'value']]
    data.time = pd.to_datetime(data.time)
    data = data.set_index('time').between_time(*time_range).reset_index()
    data.time = data.time.dt.date
    data = data.groupby(['id', 'time'])['value'].sum().reset_index()
    data['week'] = compute_week_number(data.time)
    data['location'] = name
    return data


def get_body_temperature(data):
    data = data[data.location == 'body_temperature'][['id', 'time', 'value']]
    data.time = pd.to_datetime(data.time).dt.date
    data = data.groupby(['id', 'time'])['value'].mean().reset_index()
    data['week'] = compute_week_number(data.time)
    data['location'] = 'body_temperature'
    return data


def get_bathroom_delta(data, func, name):
    data.time = pd.to_datetime(data.time).dt.date
    results = {}
    for p_id in data.id.unique():
        p_data = func(data[data.id == p_id], single_location='bathroom1',
                      recall_value=feature_config['bathroom_urgent']['recall_value'])
        if len(p_data) > 0:
            results[p_id] = p_data
    results = pd.DataFrame([(i, j, results[i][j].astype(float)) for i in results for j in results[i]],
                           columns=['id', 'time', 'value'])
    results['location'] = name
    return results


def get_weekly_activity_data(data):
    data.time = pd.to_datetime(data.time).dt.date
    data = data.groupby(['id', 'time', 'location'])['value'].sum().reset_index()
    data['week'] = compute_week_number(data.time)
    data = data[data.location.isin(config['activity']['sensors'])]
    return data
