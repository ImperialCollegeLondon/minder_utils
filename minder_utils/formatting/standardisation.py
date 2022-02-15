import pandas as pd
import numpy as np
import os
import importlib.resources as pkg_resources
from importlib.machinery import SourceFileLoader
from minder_utils.configurations import data_path, config


# import python function from path:
path_dir = data_path
with open(path_dir, 'r') as file_read:
    path = file_read.read()
dri_data_util_validate = SourceFileLoader('dri_data_util_validate', path + '/validated_date.py').load_module()
from dri_data_util_validate import validated_date


def standardise_activity_data(df):
    df = df.drop_duplicates()
    df.time = pd.to_datetime(df.time)
    df.time = pd.to_datetime(df.time.dt.strftime("%Y-%m-%d %H:%M:%S"))

    df_start = df[['id', 'time', 'location']].drop_duplicates()
    df_end = df_start.copy()
    df_start['hour'] = '00:00:00'
    df_end['hour'] = '23:00:00'
    df_borders = pd.concat([df_start, df_end])
    df_borders['time'] = pd.to_datetime(df_borders.time.dt.strftime('%Y-%m-%d')
                                                    + ' ' + df_borders.hour)
    df_borders.drop('hour', inplace=True, axis=1)

    df = df.append(df_borders, sort=False, ignore_index=True) \
        .drop_duplicates(subset=['id', 'time', 'location'])
    df = df.fillna(0).groupby(['id', 'location']).apply(lambda x: x.set_index('time')
                                                              .resample('H').sum()).reset_index()
    table_df = df.pivot_table(index=['id', 'time'], columns='location',
                              values='value').reset_index()
    table_df = table_df.replace(np.nan, 0)
    for sensor in config['activity']['sensors']:
        if sensor not in table_df.columns:
            table_df[sensor] = 0
    
    # with open(path_dir, 'r') as file_read:
    #     uti_folder_path = file_read.read()

    # patient_data = pd.read_csv(uti_folder_path + 'UTIs-TP-TN.csv')
    # patient_data = patient_data[['subject', 'datetimeCreateddf', 'valid']].dropna().drop_duplicates()
    # patient_data.columns = ['id', 'time', 'valid']
    # patient_data.id = map_raw_ids(patient_data.id, True)
    # p_data = []
    # d = validated_date()
    # for idx, p_id in enumerate(d.keys()):
    #     for data in d[p_id]:
    #         p_data.append([p_id, data[0], data[1]])
    # p_data = pd.DataFrame(p_data, columns=['id', 'time', 'valid'])
    #
    # patient_data = pd.concat([patient_data, p_data])
    #
    # patient_data.time = pd.to_datetime(pd.to_datetime(patient_data.time).dt.date)
    # patient_data['time'] = patient_data.time.dt.strftime('%Y-%m-%d') + patient_data['id'].astype(str)

    # table_df.id = map_raw_ids(table_df.id, True)
    # table_df['valid'] = table_df.time.dt.strftime('%Y-%m-%d') + table_df['id'].astype(str)
    # table_df['valid'] = table_df['valid'].map(
    #     patient_data.loc[:, ['valid', 'time']].set_index('time')['valid'].to_dict())
    table_df = table_df.dropna()
    return table_df


def standardise_physiological_environmental(df, date_range, shared_id=None):
    if shared_id is not None:
        df = df[df.id.isin(shared_id)]
    df.time = pd.to_datetime(df.time)
    df = df.groupby(['id', 'location']).apply(lambda x: x.set_index('time')
                                                                  .resample('D').mean()).reset_index().fillna(0)
    df.time = pd.to_datetime(df.time).dt.date
    idx = pd.MultiIndex.from_product((df.id.unique(), date_range, df.location.unique()), names=['id', 'time', 'location'])
    return df.set_index(['id', 'time', 'location']).reindex(idx, fill_value=0)\
        .reset_index().pivot_table(index=['id', 'time', 'location'], values='value')


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)
