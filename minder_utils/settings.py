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


def set_delta(delta=1):
    '''
    This function allows you to save the delta. This is the number 
    of days before today that you want the data to go up until.

    Arguments
    ---------

    - delta: integer:
        This is the number of days before the current date, which will act as
        the latest date downloaded in the dataset. After running this function,
        to save the dates for other classes to access, please run the 
        ```minder_utils.settings.dates_save``` function.
        Default: ```1```.

    '''
    with open(delta_path, 'w') as file_write:
        file_write.write(str(delta))
    print('Delta Saved')
    return


def dates_save(refresh=False):
    '''
    This function saves the date range that you want to download the data for.
    When you have run ```minder_utils.settings.set_delta```, then you 
    would need to run this function to update the dates used in the weekly loading
    classes.

    This function allows you to either refresh all of the data up until your ```delta```,
    or to use the dates of the data that was previously downloaded. For example, if you
    wanted to download all of the data again, you would use ```refresh=True```, but 
    if you wanted to only download the data between your last download settings and 
    now, you would use ```refresh=False```.

    Arguments
    ---------

    - refresh: bool:
        If ```True```, the currently saved settings will be overwritten and 
        the package will be ready to refresh all of the data. If ```False```, 
        the previously saved settings will be used. This allows you to fill in the 
        data between the last time you ran the code and now.
        Default: ```False```.

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


def set_data_dir(path='./data/'):
    '''
    This allows you to set the path to mappings.json, Patients.csv and UTIs-TP-TN.csv.

    Arguments
    ---------

    - path: string:
        Please supply a string value that contains the relative path from your current
        working directory to the folder containing the data.
        Default: ```'./data/'```.


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
