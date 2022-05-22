# Download and process the DRI data

For documentation and installation instructions, please see: [minder_utils documentation](https://minder-utils.github.io).


# Download and process the DRI data

Install anaconda, and create environment via:
```
# Linux
conda create --name minder_utils --file conda-linux-64.lock
# Windows
conda create --name minder_utils --file conda-win-64.lock
# Mac
conda create --name minder_utils --file conda-osx-64.lock
# M1 Mac
conda create --name minder_utils --file conda-osx-arm64.lock
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

Please share your ideas/code of formatting the data (activity, environmental, physiological, questionary), data visualisation or any other ideas with us. We will organise the code and share to others.

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
