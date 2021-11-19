import pandas as pd


def compute_week_number(df):
    df = pd.to_datetime(df)
    return df.dt.isocalendar().week + (df.dt.isocalendar().year - 2000) * 100


def frequencies_tp(input_df, tp = 3, return_locations = True):
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
    out = pd.concat([input_df, one_hot], axis = 1)
    out['time'] = pd.to_datetime(out['time'])
    out = out.groupby([pd.Grouper(key = 'time', freq='{}h'.format(tp), 
                       origin = 'start_day', 
                       dropna = False)])[one_hot.columns].sum().reset_index().fillna(0)

    if return_locations:
        return out, list(one_hot.columns)
    
    else:
        return out