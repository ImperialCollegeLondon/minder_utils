import pandas as pd
import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta


def compute_week_number(df):
    df = pd.to_datetime(df, utc=True, infer_datetime_format=True)
    return df.dt.isocalendar().week + (df.dt.isocalendar().year - 2000) * 100


def datetime_to_time(datetimes):
    '''
    This function simply conerts date time strings to just times.
    This can be useful when we want to plot histograms of average
    events on a daily basis.

    Arguments
    ---------
    datetimes: iterable
       This is an iterable list, array or series of strings
       of datetimes. This must be understandable by 
       dt.datetime.time().


    Returns
    --------
    times: DatetimeIndex
        This is a series contaning the new times. They will all
        have date value of 1900-01-01.

    ''' 
    times = [dt.datetime.time(d) for d in datetimes]
    times = pd.to_datetime(times, format="%H:%M:%S")
    
    return times

def datetime_to_day(datetimes):
    '''
    This function simply conerts date time strings to just days.
    This can be useful when we want to plot histograms of average
    events on a daily basis.

    Arguments
    ---------
    datetimes: iterable
       This is an iterable list, array or series of strings
       of datetimes. This must be understandable by 
       dt.datetime.date().


    Returns
    --------
    days: DatetimeIndex
        This is a series contaning the new days.

    ''' 
    days = [dt.datetime.date(d) for d in datetimes]
    days = pd.to_datetime(days, format="%Y-%m-%d")
    
    return days



def week_to_date(df: pd.DataFrame, day_of_week: int = 1):
    '''
    Calculate the date according to the week index. Note the week index is calcualted
    according to function ```compute_week_number```
    Args:
        df: Dataframe, a panda series contains the week index, e.g. fe.activity.week
        day_of_week: int, the index of the day of the week, e.g. Monday = 1

    Returns: Dataframe, a panda series contains the dates

    '''
    def cal_weeks(week_idx):
        year = 2000 + int(week_idx / 100)
        week = week_idx - int(week_idx / 100) * 100
        return pd.to_datetime('{}-{}-{}'.format(year, week, day_of_week), format='%Y-%W-%w')
    return df.apply(cal_weeks)


def frequencies_tp(input_df, tp=3, return_locations=True):
    '''
    Arguments
    ---------

    - input_df: pandas dataframe:
        This input. This must contain the columns ```'time'``` and ```'location'```.

    - tp: int:
        This is the number of hours the data will be grouped by.

    - return_locations: bool:
        This dictates whether the individual locations will be returned
        as a list along with the dataframe.


    Returns
    ----------
    - out: pandas dataframe:
        This dataframe contains the one hot coded ```'locations'``` column, 
        summed over ```tp```.

    '''

    one_hot = pd.get_dummies(input_df['location'])
    out = pd.concat([input_df, one_hot], axis=1)
    out['time'] = pd.to_datetime(out['time'])
    out = out.groupby([pd.Grouper(key='time', freq='{}h'.format(tp),
                                  origin='start_day',
                                  dropna=False)])[one_hot.columns].sum().reset_index().fillna(0)

    if return_locations:
        return out, list(one_hot.columns)

    else:
        return out
