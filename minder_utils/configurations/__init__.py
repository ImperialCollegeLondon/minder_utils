import os
from pathlib import Path
import yaml

p = Path(os.path.join(os.path.dirname(__file__), 'confidential/'))
if not p.exists():
    os.mkdir(p)

data_path = os.path.join(os.path.dirname(__file__), 'confidential/data_path.txt')
token_path = os.path.join(os.path.dirname(__file__), 'confidential/token_real.json')
dates_path = os.path.join(os.path.dirname(__file__), 'confidential/dates.json')
delta_path = os.path.join(os.path.dirname(__file__), 'confidential/delta.txt')
tihm_data_path = os.path.join(os.path.dirname(__file__), 'confidential/tihm_data_path.txt')

config = yaml.load(open(os.path.join(os.path.dirname(__file__), "config_dri.yaml"), "r"), Loader=yaml.FullLoader)
feature_config = yaml.load(open(os.path.join(os.path.dirname(__file__), "config_engineering_feature.yaml"), "r")
                           , Loader=yaml.FullLoader)
