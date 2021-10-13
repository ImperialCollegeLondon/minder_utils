import pandas as pd
import os
from .map_utils import map_numeric_ids, map_url_to_flag
from importlib.machinery import SourceFileLoader
from ..download.download import Downloader


# import python function from path:
path_dir = os.path.join(os.path.dirname(__file__), 'data_path.txt')
with open(path_dir, 'r') as file_read:
    path = file_read.read()
dri_data_util_validate = SourceFileLoader('dri_data_util_validate', path + '/validated_date.py').load_module()
from dri_data_util_validate import validated_date


def label_dataframe(unlabelled_df):
    """
    unlabelled_df: contains id, time but no label information
    """
    unlabelled_df['valid'] = unlabelled_df.id.astype(str) + unlabelled_df.time.dt.date.astype(str)
    
    try:
        df = pd.read_csv('./data/raw_data/procedure.csv')
    except FileNotFoundError:
        Downloader().export(categories=['procedure'])
        df = pd.read_csv('./data/raw_data/procedure.csv')
    
    df.notes = df.notes.apply(lambda x: str(x).lower())
    df = df[df.notes.str.contains('urinalysis') | df.notes.str.contains('uti') | df.notes.str.contains('positive')|df.notes.str.contains('negative')]
    df = df[['patient_id', 'start_date', 'outcome']]
    df.columns = ['patient id', 'date', 'valid']
    df.valid = map_url_to_flag(df.valid)
    df.date = pd.to_datetime(df.date).dt.date
    manual_label = validated_date(True)
    manual_label['patient id'] = map_numeric_ids(manual_label['patient id'], True)
    label_df = pd.concat([manual_label, df])
    label_df = label_df.drop_duplicates()
    label_df['time'] = label_df['patient id'].astype(str) + label_df['date'].astype(str)
    mapping = label_df[['time', 'valid']].set_index('time').to_dict()['valid']
    unlabelled_df['valid'] = unlabelled_df['valid'].map(mapping)
    return unlabelled_df


