# Download and process the DRI data


## Installation Instructions:

This was tested on python 3.8.11.

Currently tested on MacOS with M1 silicon. If you use M1 silicon, you must use the conda environment and the miniconda environment in step 1.

I have not included tensorflow, pytorch and scikit-learn within the ```setup.py``` because they all have OS unique installation instructions.


1. Firstly, if attempting to install this package on MacOS with M1 silicon then please follow the instructions here: [Installing Tensorflow on MacOS](https://github.com/apple/tensorflow_macos/issues/153) (Please ask Alex if you need help because this is quite unfriendly). Then you may move to step 3. If you are installing on windows, move to step 2.

2. Install tensorflow 2 using the command here: [Tensorflow Installation Guide](https://www.tensorflow.org/install)
    - Tested with tensorflow 2.6.0 on windows and 2.4.0 on MacOS.

3. Install pytorch, using the command here: [Pyorch Installation Guide](https://pytorch.org/get-started/locally/)
    - Tested with pytorch 1.9.1 with cuda 10.2 on windows and 1.9.1 on MacOS.

4. Install scikit-learn using the command here: [Scikit-Learn Installation Guide](https://scikit-learn.org/stable/install.html)
    - Tested with scikit-learn 1.0 on windows and 1.0 on MacOS.

4. Run ```pip install -e git+https://github.com/ImperialCollegeLondon/minder_utils.git@package#egg=minder_utils```. There is a notebook, [Install Example.ipynb](./Install%20Example.ipynb) with an example of this running in a jupyter notebook. This will save the package in the working directory. Any changes then made to this code will be reflected in your installation.


Please let me know if you run into any issues!

A getting started guide exists here: [Getting Started.ipynb](./Getting%20Started.ipynb).


## Troubleshooting:

There are many issues that arise from incompatibilities with Apple's M1 silicon.

- If you are a MacOS M1 user and ran ```conda install jupyterlab``` and it won't work when you run ```jupyter lab```. You also need to run: ```conda install nbclassic==0.2.8```.
- If scikit-learn keeps erroring, uninstall it and then do the following:
    - ``` conda install scikit-learn```
    - ``` conda install scipy```
- After installing the packages on Windows 10, I had to install ```six==1.15.0```, ```typing-extensions==3.7.4``` and ```scipy```, because tensorflow was erroring. 



# Download and process the DRI data

Install anaconda, and create environment via:
```
conda env create -f environment.yml
```
 
## Overview
NOTE: the time in the dataframe is UTC, which in the summer is 1 hour earlier then local patient time.
 1. access the research portal and activate an access token
 2. Copy and paste your token into [Getting Started.ipynb](./Getting%20Started.ipynb).

Currently, the script can
 1. Download the data
 2. Categorize the data ```python main.py -formatting True```. Will return an object with the following attributes
  - physiological_data, the values will be averaged by date.
  - activity_data
  - environmental_data, the values will be averaged by date.

The weekly_loader in the 'scripts' folder supports download the activity data weekly, it can
 - download all the previously collected activity data
 - download the latest activity data in the last week
 - put the data in a specific format
 - normalise and save the data

Please check the [Instruction.ipynb](./Instruction.ipynb) for usage.

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
from minder_utils.scirpts.weekly_loader import Weekly_dataloader
loader = Weekly_dataloader()
loader.initialise()
```
 2. Reload the latest weekly data
```
loader.refresh()
```
 3. To access the data, you can use the following script
```
unlabelled = np.load(os.path.join(loader.previous_data, 'unlabelled.npy'), allow_pickle=True)
X = np.load(os.path.join(loader.previous_data, 'X.npy'), allow_pickle=True)
y = np.load(os.path.join(loader.previous_data, 'y.npy'), allow_pickle=True)

weekly_data = np.load(os.path.join(loader.current_data, 'unlabelled.npy'), allow_pickle=True)
p_ids = np.load(os.path.join(loader.current_data, 'patient_id.npy'), allow_pickle=True)
dates = np.load(os.path.join(loader.current_data, 'dates.npy'), allow_pickle=True)
```

Currently the data will be formated into (N * 3 * 8 * 18), where N is the number of samples, 3 * 8 is 24 hours per day (the sensor readings will be aggregated hourly), 19 is the number of sensors. The 19 sensors are:
```
['WC1', 'back door', 'bathroom1', 'bedroom1',
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
