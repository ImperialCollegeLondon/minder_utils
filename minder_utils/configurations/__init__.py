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
        'columns': ['id', 'time', 'location', 'value'],
        'sensors': ['body_mass_index',
                    'body_muscle_mass',
                    'body_temperature',
                    'body_weight',
                    'diastolic',
                    'heart_rate',
                    'oxygen_saturation',
                    'skin_temperature',
                    'systolic',
                    'total_body_fat',
                    'total_body_water',
                    'total_bone_mass'],
    },
    'activity': {
        'type': ['raw_door_sensor', 'raw_appliance_use', 'raw_activity_pir'],
        'columns': ['id', 'time', 'location', 'value'],
        # 'sensors': ['WC1', 'back door', 'bathroom1', 'bedroom1',
        #             'conservatory', 'dining room', 'fridge door', 'front door',
        #             'hallway', 'kettle', 'kitchen', 'living room',
        #             'lounge', 'main door', 'microwave', 'office'],
        'sensors': ['back door', 'bathroom1', 'bedroom1', 'dining room',
                    'fridge door', 'front door', 'hallway', 'kettle', 'kitchen',
                    'living room', 'lounge', 'microwave', 'study', 'toaster']
    },
    'environmental': {
        'type': ['raw_ambient_temperature', 'raw_light'],
        'columns': ['id', 'time', 'location', 'value'],
        'sensors': ['back door', 'bathroom1', 'bedroom1', 'fridge door', 'front door',
                    'hallway', 'kitchen', 'lounge', 'study', 'conservatory',
                    'dining room', 'main door', 'cellar', 'living room', 'WC1',
                    'corridor1', 'secondary', 'office', 'garage']
    },
    'individuals': {
        'text': ['homes', 'raw_sleep_mat', 'raw_behavioural',
                 'procedure', 'observation_notes', 'encounter',
                 'issue'],
        'measure': ['raw_sleep_event', 'raw_wearable_walking']
    }
}
