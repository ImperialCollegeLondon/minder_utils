import json
import pandas as pd
import datetime as DT
from minder_utils.configurations import data_path, dates_path, token_path, delta_path


def token_save(token):
    '''
    This permanently saves your token within the package. This means that you only need 
    to supply your token once.

    Arguments
    ---------

    - token: string:
        This is the user token that can be used to access minder.

    '''
    token_dict = {"token": "Bearer {}".format(token)}
    with open(token_path, 'w') as file_write:
        json.dump(token_dict, file_write)

    print('Token Saved')

    return


def set_delta(delta):
    with open(delta_path, 'w') as file_write:
        file_write.write(str(delta))
    print('Delta Saved')
    return


def dates_save(refresh=False):
    '''
    '''
    with open(delta_path, 'r') as file_read:
        delta = file_read.read()
    delta = int(delta)
    today = DT.date.today() - DT.timedelta(days=delta)
    if refresh:
        date_dict = {'previous': {'since': None, 'until': today - DT.timedelta(days=7)},
                     'current': {'since': today - DT.timedelta(days=7), 'until': today},
                     'gap': {'since': None, 'until': None}}
    else:
        with open(dates_path) as json_file:
            date_dict = json.load(json_file)
        for state in date_dict:
            for time in date_dict[state]:
                date_dict[state][time] = pd.to_datetime(date_dict[state][time])
        date_dict['gap']['since'] = date_dict['current']['until']
        date_dict['gap']['until'] = today - DT.timedelta(days=7)
        date_dict['previous']['until'] = today - DT.timedelta(days=7)
        if today - DT.timedelta(days=7) > date_dict['current']['until']:
            date_dict['current']['since'] = today - DT.timedelta(days=7)
        else:
            date_dict['current']['since'] = date_dict['current']['until']
        date_dict['current']['until'] = today
    with open(dates_path, 'w') as file_write:
        json.dump(date_dict, file_write, default=str)

    print('Dates Saved')

    return


def set_data_dir(path):
    '''
    This allows you to set the path to mappings.json, Patients.csv and UTIs-TP-TN.csv.

    Arguments
    ---------

    - path: string:
        Please supply a string value that contains the relative path from your current
        working directory to the folder containing the data.
        For example: './data/'.


    '''

    def ensure_folder(path):
        if path[-1] != '/':
            path = path + ('/')
        return path

    path = ensure_folder(path)

    with open(data_path, 'w') as file_write:
        file_write.write(path)

    print('Path Saved')

    return
