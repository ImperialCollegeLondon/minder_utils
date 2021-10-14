# Download and process the DRI data

 
**Usage**
 1. access the research portal and activate an access token
 2. Copy and paste your token into download/token_real.json, using download/token.json as a template.
 3. run ```python main.py```

Currently, the script can
 1. Download the data ```python main.py -download True``` (Imperial VPN and access token is necessary).
 2. Categorize the data ```python main.py -formatting True```. Will return an object with the following attributes
  - physiological_data, the values will be averaged by date.
  - activity_data
  - environmental_data, the values will be averaged by date.

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
