import pandas as pd
import os
from minder_utils.formatting.map_utils import map_random_ids
from minder_utils.util.decorators import load_save
from minder_utils.configurations import config, tihm_data_path


@load_save(os.path.join('./data', 'pkl', 'raw_data'), 'TIHM')
def format_tihm_data():
    '''
    This function will change the TIHM data to the same format as DRI
    Args:
        path_to_tihm: string, Path to TIHM folder

    Returns:
        df: Dataframe, same as pir sensors in DRI data
    '''
    with open(tihm_data_path, 'r') as file_read:
        path_to_tihm = file_read.read()

    def add_location(df, location_name):
        df['location'] = location_name
        df = df[['subject', 'datetimeObserved', 'location', 'valueQuantity']]
        df.columns = config['physiological']['columns']
        return df
    sensors = {'Kettle': 'kettle', 'Bedroom': 'bedroom1', 'Kitchen': 'kitchen',
               'Bathroom': 'bathroom1', 'Hallway': 'hallway',
               'Fridge Door': 'fridge door', 'Front Door': 'front door',
               'Lounge': 'lounge', 'Back Door': 'back door', 'Toaster': 'toaster',
               'Microwave': 'microwave', 'Study': 'study',
               'Dining Room': 'dining room', 'Living Room': 'living room'}
    data = pd.read_csv(os.path.join(path_to_tihm, 'Observations.csv'))
    # deleting rows that have times and dates before the year 2000
    data = data[(pd.to_datetime(data.datetimeObserved) > pd.to_datetime('2000-01-01 00:00:00'))]
    #data.subject = map_random_ids(data.subject, True)
    activity_data = data[data.location.isin(list(sensors.keys()))] \
        [['subject', 'datetimeObserved', 'location', 'valueQuantity']]
    body_temperature = data[data.device == 27991004] \
        [['subject', 'datetimeObserved', 'valueQuantity']]
    blood_pressure = data[data.device == 70665002] \
        [['subject', 'datetimeObserved', 'valueQuantity']]
    scale = data[data.device == 19892000] \
        [['subject', 'datetimeObserved', 'valueQuantity']]

    activity_data.valueQuantity = 1
    activity_data.columns = config['activity']['columns']

    body_temperature = add_location(body_temperature, 'body_temperature')
    blood_pressure = add_location(blood_pressure, 'blood_pressure')
    scale = add_location(scale, 'scale')
    physiological = pd.concat([body_temperature, blood_pressure, scale])

    activity_data.location = activity_data.location.map(sensors)
    activity_data.time = pd.to_datetime(activity_data.time, utc=True)
    physiological.time = pd.to_datetime(physiological.time, utc=True)
    data = {'activity': activity_data,
            'physiological': physiological}
    return data

