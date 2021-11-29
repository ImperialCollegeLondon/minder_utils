from minder_utils.models.feature_extractors import SimCLR, Partial_Order, AutoEncoder
from minder_utils.dataloader import process_data
from minder_utils.evaluate.evaluate_models import evaluate_features
from minder_utils.dataloader import Dataloader
from minder_utils.util.initial import first_run
import pandas as pd
import numpy as np
import os

os.chdir('..')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# If you haven't processed the data, please uncomment the line below
# first_run()

# if you have processed the data, then just pass None to the dataloader
labelled_data = Dataloader(None).labelled_data
unlabelled_data = Dataloader(None).unlabelled_data['activity']

unlabelled, X, y, p_ids = process_data(labelled_data, unlabelled_data, num_days_extended=0)

# Train feature extractors and save the features
# Note you only need to train them once, then the model will be saved automatically.
# If you want to retrain the model, set retrain as True in config_feature_extractor.yaml
# SimCLR().fit(unlabelled).transform(X)
AutoEncoder().fit(unlabelled).transform(X)

# All the features have been saved and it's ready to test
df = evaluate_features(X, y, p_ids, num_runs=10, valid_only=True)
df['extractor'] = 'None'
results = [df]
for feature in ['simclr', 'autoencoder']:
    feat = np.load('./data/extracted_features/{}.npy'.format(feature))
    df = evaluate_features(feat, y, p_ids, num_runs=10, valid_only=True)
    df['extractor'] = feature
    results.append(df)
print(pd.concat(results))
