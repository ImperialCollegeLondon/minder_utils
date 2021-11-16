import numpy as np
import pandas as pd
from .DensityFunctions import BaseDensityCalc


def raw_delta_calc(times):
    '''
    Given an array of times, this function calculates the deltas between them.

    Arguments
    ---------
    
    - times: array:
        This is an array of times that will be used to calculate the deltas.

    Returns
    ---------

    - out: array:
        This is an array of deltas.

    '''

    out = (times[1:] - times[:-1])*1e-9

    out = out.astype(float)


    return out



def single_location_delta(input_df, single_location, 
                            columns={'time': 'time', 'location': 'location'}, recall_value=5, 
                            return_as_list = False):
    '''
    This function takes the ```input_df``` and calculates the raw time delta between the single_location location time
    and the time of the ```recall_value``` number of locations immediately before the single_location.

    This does not separate on subject. Please pass data from a single subject into this function.

    Arguments
    ---------

    - input_df: pandas dataframe:
        This is a dataframe that contains columns relating to the subject, time and location of sensor trigger.

    - single_location: string:
        This is the location value that you wish to calculate the delta to.
    
    - columns: dictionary:
        This is the dictionary with the column names in ```input_df``` for each of the values of data that we need 
        in our calculations.
        This dictionary should be of the form:
        ```
        {'time':      column containing the times of sensor triggers,
         'location':  column containing the locations of the sensor triggers}
         ```

    - recall_value: integer:
        This is the number of previous locations to the single_location trigger

    - return_as_list: bool:
        This option allows the user to return a list of the dates and data if ```True```. This is 
        used internally by other functions.

    
    Returns
    ---------

    - out: dictionary:
        This has the Timestamps of the dates as keys (for example: Timestamp('2021-05-05 00:00:00')) and the 
        arrays of deltas as values. The arrays of deltas are of shape ```(Nt, recall_value)``` where Nt is the 
        number of visits to ```single_location``` on a given day. If there are no ```single_location``` visits
        found in the data, then an empty dictionary will be returned.

    '''
    time_column = columns['time']
    location_column = columns['location']

    # format the incoming data to ensure assumptions about structure are met
    input_df[time_column] = pd.to_datetime(input_df[time_column], utc=True)
    input_df = input_df.sort_values(time_column)

    # find the indices of the data that match with the location we want to find the delta to
    single_location_indices = np.where(input_df[location_column] == single_location)[0].reshape(-1, 1)
    # making sure that the recall value is not more than the number of sensor triggers before the
    # first single_location sensor trigger
    if len(single_location_indices) ==  0:
        
        if return_as_list: return [], []
        else: return {}

    single_location_indices = single_location_indices[np.argmax(recall_value < single_location_indices):]

    # indices of the sensor triggers that we need in our calculations
    recall_indices = np.hstack([single_location_indices - i for i in range(recall_value + 1)])

    # the times of the sensor triggers
    recall_times = input_df[time_column].values[recall_indices]

    # the delta between the times for each of the previous sensors to recall_value
    recall_delta = (recall_times[:, 0, None] - recall_times[:, 1:]) * 1e-9

    # the times of the single_location triggers
    single_location_times = input_df[time_column].iloc[single_location_indices.reshape(-1, )]
    # dates of the single_location triggers
    single_location_dates = single_location_times.dt.date

    # out dictionary
    out = {}


    if return_as_list:
        date_list = []
        data_list = []
        for nd, date in enumerate(single_location_dates.unique()):
            date_list.append(date)
            data_to_add = recall_delta[single_location_dates.values == date].astype(float)
            data_list.append(data_to_add)
        
        return pd.to_datetime(date_list), data_list


    else:
        # creating the output dictionary
        for date in single_location_dates.unique():
            # saving the delta values for this date to the dictionary
            out[pd.to_datetime(date)] = recall_delta[single_location_dates.values == date].astype(float)

        return out


class TimeDeltaDensity(BaseDensityCalc):
    '''
    This function allows the user to calculate reverse percentiles on some data, given another
    dataset.

    '''

    def __init__(self, save_baseline_array=True, sample=False, sample_size=10000,
                 seed=None, verbose=True):
        BaseDensityCalc.__init__(self, save_baseline_array=save_baseline_array,
                                 sample=sample, sample_size=sample_size, seed=seed, verbose=verbose)

        return




def rp_single_location_delta(input_df, single_location, baseline_length_days = 7, baseline_offset_days = 0,
                             columns={'time': 'time', 'location': 'location'}, recall_value=5):
    '''
    This function takes the ```input_df``` and calculates the reverse percentage time delta between the ```single_location``` location time
    and the time of the ```recall_value``` number of locations immediately before the ```single_location```. The baseline
    for the rever percentage calculation is defined by ```baseline_length_days``` and ```baseline_offset_days```. 

    For example:
    With ```baseline_length_days = 7``` and ```baseline_offset_days = 1```, the rp deltas on the day
    ```pd.Timestamp('2021-06-29')``` are calculated using the deltas from 
    ```pd.Timestamp('2021-06-21 00:00:00')``` to ```pd.Timestamp('2021-06-28 00:00:00')```.

    This does not separate on subject. Please pass data from a single subject into this function.

    Arguments
    ---------

    - input_df: pandas dataframe:
        This is a dataframe that contains columns relating to the subject, time and location of sensor trigger.

    - single_location: string:
        This is the location value that you wish to calculate the delta to.
    
    - baseline_length_days: integer:
        This is the length of the baseline in days that will be used. This value is used when finding
        the ```baseline_length_days``` complete days of ```single_location``` data to use as a baseline.
    
    - baseline_offset_days: integer:
        This is the offset to the baseline period. ```0``` corresponds to a time period ending the morning of the
        current date being calculated on.
    
    - columns: dictionary:
        This is the dictionary with the column names in ```input_df``` for each of the values of data that we need 
        in our calculations.
        This dictionary should be of the form:
        ```
        {'time':      column containing the times of sensor triggers,
         'location':  column containing the locations of the sensor triggers}
         ```

    - recall_value: integer:
        This is the number of previous locations to the single_location trigger
    
    
    Returns
    ---------

    - out: dictionary:
        This has the Timestamps of the dates as keys (for example: Timestamp('2021-05-05 00:00:00')) and the 
        arrays of deltas as values. The arrays of deltas are of shape ```(Nt, recall_value)``` where Nt is the 
        number of visits to ```single_location``` on a given day.

    '''


    # column names
    time_column = columns['time']
    location_column = columns['location']

    out = {}

    # format the incoming data to ensure assumptions about structure are met
    input_df[time_column] = pd.to_datetime(input_df[time_column], utc=True)
    input_df = input_df.sort_values(time_column)

    # getting the single location raw delta
    date_list, data_list = single_location_delta(input_df, single_location, columns, recall_value, return_as_list=True)

    # for each date
    for nd, date in enumerate(date_list):
        date = pd.to_datetime(date)


        '''
        if len(baseline_offset)>0:
            baseline_start_tp = pd.to_datetime(date - pd.Timedelta(**baseline_length) - pd.Timedelta(**baseline_offset))
            baseline_end_tp = pd.to_datetime(date - pd.Timedelta(**baseline_offset))
        else:
            baseline_start_tp = pd.to_datetime(date - pd.Timedelta(**baseline_length))
            baseline_end_tp = pd.to_datetime(date)
        '''

        

        index_baseline_end = np.where(date_list <= date)[0][-1]
        
        index_baseline_end = index_baseline_end - baseline_offset_days
        index_baseline_start = index_baseline_end - baseline_length_days

        if index_baseline_start < 0:
            out[date] = -1*np.ones_like(data_list[nd])
            continue

        
        baseline_delta = np.vstack([data_list[index] for index in range(index_baseline_start, index_baseline_end)])

        
        td = TimeDeltaDensity(save_baseline_array=True, sample=True, sample_size=10000,
                                seed=None, verbose=False)

        td.fit(baseline_delta)
        out[date] = td.transform(data_list[nd])
        
    return out



def datetime_to_clock(times):
    '''
    This function converts each date time in an array to a vector
    of size 2 which represents the time in a continuous way.  The vector represents 
    the co-ordinates of a cirlce for which the time would represent on a 24-hour analogue clock.
    
    Arguments
    ---------
    
    - times: array:
        This is an array of times to be converted. The shape of this 
        array should be (N,) or (N,1).
    
    Returns
    ---------
    
    - out: array:
        This is an array containing the transformed times. This array
        will be of size (N,2). The vector represents the co-ordinates of 
        a cirlce for which the time would represent on a 24-hour analogue clock.
    
    '''

    times = pd.to_datetime(times, utc=True)

    total_seconds = times.hour * 3600 \
                    + times.minute * 60 \
                    + times.second \
                    + 1e-6 * times.microsecond
    total_seconds = np.asarray(total_seconds)

    C = 24 * 3600
    x = (np.sin(2 * np.pi * total_seconds / C) + 1e-12).reshape(-1, 1)
    y = (np.cos(2 * np.pi * total_seconds / C) + 1e-12).reshape(-1, 1)

    out = np.hstack([x, y])

    return out
