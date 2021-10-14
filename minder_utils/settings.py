import json
import importlib.resources as pkg_resources
import os


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
    token_dir = os.path.join(os.path.dirname(__file__), 'download', 'token_real.json')
    with open(token_dir, 'w') as file_write:
        json.dump(token_dict, file_write)

    print('Token Saved')

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

    file_dir = os.path.join(os.path.dirname(__file__), 'formatting', 'data_path.txt')
    with open(file_dir, 'w') as file_write:
        file_write.write(path)

    print('Path Saved')

    return
