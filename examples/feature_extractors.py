from minder_utils.models.feature_extractors import SimCLR, Partial_Order, AutoEncoder
from minder_utils.dataloader import process_data
from minder_utils.evaluate.evaluate_models import evaluate_features
from minder_utils.dataloader import Dataloader
import pandas as pd
import numpy as np
import os

os.chdir('..')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def first_run():
    from minder_utils.formatting import Formatting
    from minder_utils.download import Downloader
    # if you haven't download the data, uncomment next line
    # Downloader().export()

    loader = Formatting()
    dataloader = Dataloader(loader.activity_data, max_days=10, label_data=True)
    # This will automatically process and save the data
    data = dataloader.labelled_data
    '''
    Note
        - this will take a while, but you don't need to do it again
          if you set the refresh as False in config_dri.yaml, will optimise it in future.
    
        - this unlabelled data contains only DRI data, if you want to include the TIHM
          data as well, set date in Dataloader as None. 
    '''
    unlabelled_data = dataloader.unlabelled_data


# If you haven't processed the data
# first_run()

# if you have processed the data, then just pass None to the dataloader
labelled_data = Dataloader(None).labelled_data
unlabelled_data = Dataloader(None).unlabelled_data['activity']

unlabelled, X, y, p_ids = process_data(labelled_data, unlabelled_data, num_days_extended=0)

# Train feature extractors and save the features
# Note you only need to train them once, then the model will be saved automatically.
# If you want to retrain the model, set retrain as True in config_feature_extractor.yaml
SimCLR().fit(unlabelled).transform(X)
AutoEncoder().fit(unlabelled).transform(X)

# This feature extractor is under testing.
_, X_extend, y_extend, _ = process_data(labelled_data, unlabelled_data, flatten=False, num_days_extended=8)
Partial_Order().fit([X_extend, y_extend]).transform(X)

# All the features have been saved and it's ready to test
df = evaluate_features(X, y, p_ids, num_runs=10, valid_only=True)
df['extractor'] = 'None'
results = [df]
for feature in ['simclr', 'autoencoder', 'partial_order']:
    feat = np.load('./data/extracted_features/{}.npy'.format(feature))
    df = evaluate_features(feat, y, p_ids, num_runs=10, valid_only=True)
    df['extractor'] = feature
    results.append(df)
print(pd.concat(results))
