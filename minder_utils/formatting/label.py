import pandas as pd
import numpy as np
from pathlib import Path
from .map_utils import map_numeric_ids, map_url_to_flag
from importlib.machinery import SourceFileLoader
from ..download.download import Downloader
from minder_utils.configurations import data_path
from minder_utils.util.util import reformat_path
import os
import datetime


def load_manual_labels():

    # import python function from path:

    with open(data_path, 'r') as file_read:
        path = file_read.read()

    path_path = Path(reformat_path(path + '/validated_date.py'))
    
    try:
        dri_data_util_validate = SourceFileLoader('dri_data_util_validate', reformat_path(path + '/validated_date.py')).load_module()
    except FileNotFoundError:
        print('Manual label file not found, you might be missing labels!!')
        return None
    
    from dri_data_util_validate import validated_date

    return validated_date




def label_dataframe(unlabelled_df, save_path='./data/raw_data/', days_either_side = 0):
    '''
    This function will label the input dataframe based on the information in ```procedure.csv``` and
    manual labels from TIHM.
    
    Arguments
    ----------
    
    - unlabelled_df: pandas dataframe:
        Unlabelled dataframe, must contain columns ```[id, time]```, where ```id``` is the
        ids of participants, ```time``` is the time of the sensors.

    - save_path: str:
        This is the path that points to the ```procedure.csv``` file. If this 
        file does not exist, it will be downloaded to this path.

    Returns
    ---------
    
    - unlabelled_df: pandas dataframe:
        This is a dataframe containing the original data along with a new column, ```'labels'```,
        which contains the labels.

    '''
    validated_date = load_manual_labels()
    save_path = reformat_path(save_path)
    try:
        df = pd.read_csv(os.path.join(save_path, 'procedure.csv'))
    except FileNotFoundError:
        Downloader().export(categories=['procedure'], save_path=save_path)
        df = pd.read_csv(os.path.join(save_path, 'procedure.csv'))

    df.notes = df.notes.apply(lambda x: str(x).lower())
    df = df[df.notes.str.contains('urinalysis') | df.notes.str.contains('uti') | df.notes.str.contains(
        'positive') | df.notes.str.contains('negative')]
    df = df[['patient_id', 'start_date', 'outcome']]
    df.columns = ['patient id', 'date', 'valid']
    df.valid = map_url_to_flag(df.valid)
    df.date = pd.to_datetime(df.date).dt.date
    df = df.dropna()
    if not validated_date is None:
        manual_label = validated_date(True)
        #manual_label['patient id'] = map_numeric_ids(manual_label['patient id'], True)
        label_df = pd.concat([manual_label, df])
        label_df = label_df.drop_duplicates()
    else:
        label_df = df.drop_duplicates()
    if not days_either_side == 0:
        def dates_either_side_group_by(x):
            date = pd.to_datetime(x['date'].values[0])
            x = [x]*(2*days_either_side+1)
            new_date_values = np.arange(-days_either_side, days_either_side + 1)
            new_dates = [date + datetime.timedelta(int(value)) for value in new_date_values]
            x = pd.concat(x)
            x['date'] = new_dates
            return x
        label_df = label_df.groupby(['patient id', 'date', 'valid']).apply(dates_either_side_group_by).reset_index(drop=True)
    label_df['time'] = label_df['patient id'].astype(str) + label_df['date'].astype(str)
    mapping = label_df[['time', 'valid']].set_index('time').to_dict()['valid']
    unlabelled_df['valid'] = unlabelled_df.id.astype(str) + unlabelled_df.time.dt.date.astype(str)
    unlabelled_df['valid'] = unlabelled_df['valid'].map(mapping)
    return unlabelled_df


def label_array(patient_ids, time, save_path='./data/raw_data/'):
    """
    This function returns labels given an array of ids and an array of times. Please see the
    following for the description of the shapes required.

    Arguments
    ---------

    - patient_ids: array:
        This is an array containing the patient IDs corresponding to the times in ```time```.
        This should be of shape (N,).

    - time: array:
        This is an array containing the times of events corresponding to the patient IDs 
        in ```patient_ids```. This should be of shape (N,). These should be of a format 
        that is acceptable by ```pandas.to_datetime()```.

    - save_path: str:
        This is the path that points to the ```procedure.csv``` file. If this 
        file does not exist, it will be downloaded to this path.

    Returns
    ---------

    - labels: array:
        This is an array containing the labels for UTIs for the given inputs.

    """
    save_path = reformat_path(save_path)
    df_dict = {'id': patient_ids, 'time': pd.to_datetime(time, utc=True)}
    unlabelled_df = pd.DataFrame(df_dict)
    unlabelled_df = label_dataframe(unlabelled_df, save_path=save_path)

    return unlabelled_df['valid'].values


def label_by_week(df):
    '''
    label the dataframe by week
    Args:
        df: Dataframe, contains columns ['id', 'week']

    Returns:

    '''
    validated_date = load_manual_labels()
    manual_label = validated_date(True)
    #manual_label['patient id'] = map_numeric_ids(manual_label['patient id'], True)
    manual_label.date = pd.to_datetime(manual_label.date)
    manual_label['week'] = manual_label.date.dt.isocalendar().week + \
                           (manual_label.date.dt.isocalendar().year - 2000) * 100

    manual_label['week'] = manual_label['patient id'].astype(str) + manual_label['week'].astype(str)
    mapping = manual_label[['week', 'valid']].set_index('week').to_dict()['valid']

    df['valid'] = df.id.astype(str) + df.week.astype(str)
    df['valid'] = df['valid'].map(mapping)
    return df