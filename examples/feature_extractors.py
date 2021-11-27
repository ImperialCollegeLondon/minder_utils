from minder_utils.models.feature_extractors import SimCLR, Partial_Order, AutoEncoder
from minder_utils.dataloader import load_data
from minder_utils.evaluate.evaluate_models import evaluate_features
import pandas as pd
import numpy as np
import os

os.chdir('..')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Load the data
unlabelled, X, y, p_ids = load_data(num_days_extended=0)

# Train feature extractors and save the features
# Note you only need to train them once, then the model will be saved automatically.
# If you want to retrain the model, set retrain as True in config_feature_extractor.yaml

SimCLR().fit(unlabelled).transform(X)
AutoEncoder().fit(unlabelled).transform(X)

# This feature extractor is under testing.
_, X_extend, y_extend, _ = load_data(flatten=False)
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