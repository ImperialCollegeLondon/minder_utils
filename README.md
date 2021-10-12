# Download and process the DRI data

Install anaconda, and create environment via:
```
conda env create -f environment.yml
```
 
## Overview
 1. access the research portal and activate an access token
 2. Copy and paste your token into download/token.json. 

Currently, the script can
 1. Download the data ```python main.py -download True``` (Imperial VPN and access token is necessary).
 2. Categorize the data ```python main.py -formatting True```. Will return an object with the following attributes
  - physiological_data, the values will be averaged by date.
  - activity_data
  - environmental_data, the values will be averaged by date.

The weekly_loader in the 'scripts' folder supports download the activity data weekly, it can
 - download all the previously collected activity data
 - download the latest activity data in the last week
 - put the data in a specific format
 - normalise and save the data


## Usage
**Download**
 1. Download the specific data
 ```
 from download.download import Downloader
 Downloader.export(since='2021-10-01', reload=True, categories=['procedure'], save_path='./data/raw_data/')
 ```
 2. Download all the data
 ```
 Downloader.export(categories='all')
 ```
 Note: If you would like to download the data you downloaded before, set the reload to False
 ```
 Downloader.export(reload=False)
 ```
**Download weekly**
Currently the script supports activity data only,
 1. First time
```
from scirpts.weekly_loader import Weekly_dataloader
loader = Weekly_dataloader(num_days_extended=5)
loader.load_data(reload_weekly=True, reload_all=False)
```
 2. Reload the latest weekly data
```
loader.load_data(reload_weekly=True, reload_all=False)
```
 3. To access the data, you can use the following script
```
# Previous data
unlabelled = np.load(os.path.join(loader.previous_data, 'unlabelled.npy')) # Unlabelled activity data
X = np.load(os.path.join(loader.previous_data, 'X.npy')) # labelled UTI
y = np.load(os.path.join(loader.previous_data, 'y.npy')) # label
label_p_ids = np.load(os.path.join(loader.previous_data, 'label_ids.npy'))

# Weekly data
weekly_data = np.load(os.path.join(loader.weekly_data, 'unlabelled.npy'))
p_ids = np.load(os.path.join(loader.weekly_data, 'patient_id.npy'))
dates = np.load(os.path.join(loader.weekly_data, 'dates.npy'))
```

Currently the data will be formated into (N * 3 * 8 * 19), where N is the number of samples, 3 * 8 is 24 hours per day (the sensor readings will be aggregated hourly), 19 is the number of sensors. The 19 sensors are:
```
['WC1', 'back door', 'bathroom1', 'bedroom1', 'cellar',
'conservatory', 'dining room', 'fridge door', 'front door',
'hallway', 'iron', 'kettle', 'kitchen', 'living room',
'lounge', 'main door', 'microwave', 'multi', 'office']

```

Please share your ideas/code of formatting the data (activity, environmental, physiological, questionary), data visualisation or any other ideas with me. I will organise the code so everyone can use it.

## TODO

**Data**
1. The activity data will be aggreated hourly
2. The missing physiological data will be imputed by mean or the nearest data.
3. Textual data will be processed by text embedding.

**Model**
1. Unsupervised learning models including autoencoder, contrastive encoder, partial order etc.
2. Classifiers including conventional classifiers, pnn
3. NLP models for processing the textual data
4. models for data fusion
