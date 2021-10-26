import os
from pathlib import Path

p = Path(os.path.join(os.path.dirname(__file__), 'confidential/'))
if not p.exists():
    os.mkdir(p)

data_path = os.path.join(os.path.dirname(__file__), 'confidential/data_path.txt')
token_path = os.path.join(os.path.dirname(__file__), 'confidential/token_real.json')
dates_path = os.path.join(os.path.dirname(__file__), 'confidential/dates.json')
delta_path = os.path.join(os.path.dirname(__file__), 'confidential/delta.txt')

config = {
    'physiological': {
        'type': ['raw_body_weight', 'raw_skin_temperature',
                 'raw_body_temperature', 'raw_body_muscle_mass',
                 'raw_heart_rate', 'raw_oxygen_saturation',
                 'raw_total_body_fat', 'raw_body_mass_index',
                 'raw_blood_pressure', 'raw_total_body_water',
                 'raw_total_bone_mass'],
        'columns': ['id', 'time', 'location', 'value']
    },
    'activity': {
        'type': ['raw_door_sensor', 'raw_appliance_use', 'raw_activity_pir'],
        'columns': ['id', 'time', 'location', 'value']
    },
    'environmental': {
        'type': ['raw_ambient_temperature', 'raw_light'],
        'columns': ['id', 'time', 'location', 'value']
    },
    'individuals': {
        'text': ['homes', 'raw_sleep_mat', 'raw_behavioural',
                 'procedure', 'observation_notes', 'encounter',
                 'issue'],
        'measure': ['raw_sleep_event', 'raw_wearable_walking']
    }
}
