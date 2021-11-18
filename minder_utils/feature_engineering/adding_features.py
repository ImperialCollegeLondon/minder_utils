from minder_utils.configurations import feature_config, config
from .util import *
import pandas as pd


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
    out.columns = ['id','value']

    out_rp = (pd.DataFrame(out.value.values.tolist())
             .stack()
             .reset_index(level=1)
             .rename(columns={0:'val','level_1':'key'}))

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
    return data
