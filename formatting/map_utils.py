import pandas as pd
import json


def map_raw_ids(p_id, df=False, path='./data/confidential'):
    """
    :param p_id:
    :param df:
    :param path: Path to the patient information, contains mappings.json and Patients.csv
    :return:
    """
    json_file_path = path + "/mappings.json"
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
    contents = {v: k for k, v in contents.items()}
    patient_ids = pd.read_csv(path + '/Patients.csv')
    patient_ids = patient_ids[['subjectId', 'sabpId']].set_index('subjectId')['sabpId'].to_dict()
    if df:
        return list(map(patient_ids.get, list(map(contents.get, p_id))))
    return int(patient_ids[contents[p_id]])


def map_numeric_ids(p_id, df=False, path='./data/confidential'):
    """
    :param p_id:
    :param df:
    :param path: Path to the patient information, contains mappings.json and Patients.csv
    :return:
    """
    json_file_path = path + "/mappings.json"
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
    # contents = {v: k for k, v in contents.items()}
    patient_ids = pd.read_csv(path + '/Patients.csv')
    patient_ids = patient_ids[['subjectId', 'sabpId']].set_index('sabpId')['subjectId'].to_dict()
    if df:
        return list(map(contents.get, list(map(patient_ids.get, p_id))))
    return contents[patient_ids[p_id]]


def map_url_to_flag(urls):
    url_mapping = {
        'http://snomed.info/sct|260385009': False,
        'http://snomed.info/sct|10828004': True,
        'http://snomed.info/sct|82334004': None,
    }

    return list(map(url_mapping.get, urls))