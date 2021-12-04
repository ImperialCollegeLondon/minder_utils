from minder_utils.configurations import feature_config, config
import numpy as np
from .calculation import entropy_rate_from_sequence
from .TimeFunctions import rp_location_delta
from .util import *
from minder_utils.util.util import PBar
import pandas as pd
from typing import Union
import sys


def get_moving_average(df:pd.DataFrame, name, w:int = 3):
    '''
    This function calculates the moving average of the values in the ```'value'``` 
    column. It will return the dataframe with the moving average in the column
    ```'value'```.
    
    
    Arguments
    ---------
    
    - df: pandas.DataFrame: 
        A dataframe containing at least a column called ```'value'```.
    
    - name: string: 
        This string is the name that will be given in the column ```'location'```.
        If this column does not already exist, it will be added.
    
    - w: int:
        The size of the moving window when calculating the moving average.
        Defaults to ```3```.
        
    Returns
    --------
    
    - df: pandas.Dataframe : 
        The original dataframe, with a new column containing the moving average. There
        will be missing values in the first ```w-1``` rows, caused by the lack of values 
        to calculate a mean using the moving window.
    
    
    '''

    
    values = df['value'].values

    if values.shape[0]<w:
        df['value'] = pd.NA
        df['location'] = name
        return df

    # moving average
    w = 3
    values_ma = np.convolve(values, np.ones(w), 'valid')/w
    df['value'] = pd.NA
    df.loc[df.index[w-1:],'value'] = values_ma
    df['location'] = name

    

    return df



def get_value_delta(df:pd.DataFrame, name):
    '''
    This function calculates the delta of the values in the ```'value'``` 
    column. It will return the dataframe with thenew values in the column
    ```'value'```.
    
    
    Arguments
    ---------
    
    - df: pandas.DataFrame: 
        A dataframe containing at least a column called ```'values'```.
    
    - name: string: 
        This string is the name that will be given in the column ```'location'```.
        If this column does not already exist, it will be added.
    
        
    Returns
    --------
    
    - df: pandas.Dataframe : 
        The original dataframe, with a new column containing the delta values. There
        will be a missing value in the first row, since delta can not be calculated here.
        
    
    
    '''

    
    values = df['value'].values

    if values.shape[0]<2:
        df['value'] = pd.NA
        df['location'] = name
        return df


    values_delta = values[1:]/values[:-1]
    df['value'] = pd.NA
    df.loc[df.index[1:],'value'] = values_delta
    df['location'] = name

    

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
    def func_group_by(x):
        x = func(input_df=x, single_location='bathroom1',
                 recall_value=feature_config['bathroom_urgent']['recall_value'])
        return x

    out = data.groupby(by=['id'])[['time', 'location']].apply(
        func_group_by).reset_index()
    out.columns = ['id', 'value']

    out_rp = (pd.DataFrame(out.value.values.tolist())
              .stack()
              .reset_index(level=1)
              .rename(columns={0: 'val', 'level_1': 'key'}))

    out = out.drop('value', 1).join(out_rp).reset_index(drop=True).dropna()
    out.columns = ['id', 'time', 'value']
    out['week'] = compute_week_number(out.time)
    out['location'] = name

    return out


def get_bathroom_delta_v1(data, func, name):
    data.time = pd.to_datetime(data.time).dt.date
    results = {}
    for p_id in data.id.unique():
        p_data = func(data[data.id == p_id].copy(), single_location='bathroom1',
                      recall_value=feature_config['bathroom_urgent']['recall_value'])
        if len(p_data) > 0:
            results[p_id] = p_data
    results = pd.DataFrame([(i, j, results[i][j].astype(float)) for i in results for j in results[i]],
                           columns=['id', 'time', 'value'])
    results['week'] = compute_week_number(results.time)
    results['location'] = name
    return results


def get_weekly_activity_data(data):
    data.time = pd.to_datetime(data.time).dt.date
    data = data.groupby(['id', 'time', 'location'])['value'].sum().reset_index()
    data['week'] = compute_week_number(data.time)
    data = data[data.location.isin(config['activity']['sensors'])]
    data = data.pivot_table(index=['id', 'week'], columns='location',
                            values='value').reset_index().replace(np.nan, 0)
    return data


def get_outlier_freq(data, func, name):
    data.time = pd.to_datetime(data.time).dt.date

    def func_group_by(x):
        x = func(x, outlier_class=feature_config['outlier_score_activity']['outlier_class'],
                 tp_for_outlier_hours=feature_config['outlier_score_activity']['tp_for_outlier_hours'],
                 baseline_length_days=feature_config['outlier_score_activity']['baseline_length_days'],
                 baseline_offset_days=feature_config['outlier_score_activity']['baseline_offset_days'])
        return x

    outlier_data = data.groupby(by=['id'])[['time', 'location']].apply(
        func_group_by).reset_index()[['id', 'time', 'outlier_score']]

    outlier_data = outlier_data.groupby(['id', pd.Grouper(key='time', freq='1d',
                                                          origin='start_day',
                                                          dropna=False)]).mean().reset_index()

    outlier_data['week'] = compute_week_number(outlier_data.time)
    outlier_data['location'] = name
    outlier_data.columns = ['id', 'time', 'value', 'week', 'location']

    return outlier_data



def get_entropy_rate(df: pd.DataFrame, sensors: Union[list, str] = 'all', name='entropy_rate') -> pd.DataFrame:
    '''
    This function allows the user to return a pandas.DataFrame with the entropy rate calculated
    for every week.
    
    
    
    Arguments
    ---------
    
    - df: pandas.DataFrame: 
        A pandas.DataFrame containing ```'id'```, ```'week'```, ```'location'```.
    
    - sensors: Union[list, str]: 
        The values of the ```'location'``` column of ```df``` that will be 
        used in the entropy calculations.
        Defaults to ```'all'```.
    
    
    
    Returns
    --------
    
    - out: pd.DataFrame : 
        This is a dataframe, in which the entropy rate is located in the ```'value'``` column.
    
    
    '''


    assert len(sensors) >= 2, 'need at least two sensors to calculate the entropy'

    # Filter the sensors
    if isinstance(sensors, list):
        df = df[df.location.isin(sensors)]
    elif isinstance(sensors, str):
        assert sensors == 'all', 'only accept all as a string input for sensors'


    df['week'] = compute_week_number(df['time'])

    def entropy_rate_from_sequence_groupby(x):
        x = entropy_rate_from_sequence(x.values)
        return x

    df = df.groupby(by=['id','week'])['location'].apply(entropy_rate_from_sequence_groupby).reset_index()
    df.columns = ['id', 'week', 'value']
    df['location'] = name

    return df



def get_subject_rp_location_delta(data, 
                                  columns = {'subject': 'patient_id', 'time': 'start_date', 'location': 'location_name'}, 
                                  baseline_length_days = 7,
                                  baseline_offset_days = 0,
                                  all_loc_as_baseline = False,
                                  name='rp_location_delta'):
    '''
    This function allows the user to calculate the rp delta for each subject and event.
    
    
    Arguments
    ---------
    
    - data: pandas dataframe:
        This is the dataframe containing the time and locations that will be used to calculate the reverse 
        percentage deltas
    
    - columns: dictionary:
        This is the dictionary with the column names in ```input_df``` for each of the values of data that we need 
        in our calculations.
        This dictionary should be of the form:
        ```
        {'subject':   column containing the subjects IDs,
         'time':      column containing the times of sensor triggers,
         'location':  column containing the locations of the sensor triggers}
        Defaults to ```{'subject': 'patient_id', 'time': 'start_date', 'location': 'location_name'}```.
    
    - baseline_length_days: integer:
        This is the length of the baseline in days that will be used. This value is used when finding
        the ```baseline_length_days``` complete days of ```single_location``` data to use as a baseline.
        Defaults to 7.
    
    - baseline_offset_days: integer:
        This is the offset to the baseline period. ```0``` corresponds to a time period ending the morning of the
        current date being calculated on.
        Defaults to 0.
    
    - all_loc_as_baseline: bool:
        This argument dictates whether all the locations are used as part of the calculation for the reverse
        percentage or if only the values from the ```to``` locations are used.
        Defaults to True.
    
    Returns
    ---------

    - out: pandas dataframe:
        This is the outputted dataframe with the rp delta values.
    
    
    '''
    subject_list = data[columns['subject']].unique()
    bar = PBar(20, len(subject_list))

    sys.stdout.write('Subject: {} {}/{}'.format(bar.give(), 0, len(subject_list)))
    sys.stdout.flush()

    def rp_location_delta_group_by(x):

        x = rp_location_delta(x, 
                        columns = columns, 
                        baseline_length_days = baseline_length_days,
                        baseline_offset_days = baseline_offset_days, 
                        all_loc_as_baseline = all_loc_as_baseline)

        bar.update(1)
        sys.stdout.write('\r')
        sys.stdout.write('Subject: {} {}/{}'.format(bar.give(), bar.progress, len(subject_list)))
        sys.stdout.flush()

        return x


    out = data.groupby(by=columns['subject']).apply(rp_location_delta_group_by).reset_index().sort_values(columns['time'])

    bar.update(1)
    sys.stdout.write('\r')
    sys.stdout.write('Subject: {} {}/{}'.format(bar.give(), bar.progress, len(subject_list)))
    sys.stdout.flush()

    out = out[[columns['subject'], columns['time'], 'from', 'to', 'rp']]

    out.columns = ['id', 'time', 'from', 'to', 'value']

    out['location'] = name
    out['week'] = compute_week_number(out['time'])


    values = out['value'].values
    values[values == -1] = np.nan
    out['value'] = values


    return out