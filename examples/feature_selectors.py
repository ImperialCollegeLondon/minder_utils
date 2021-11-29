from minder_utils.models.feature_selectors import Intrinsic_Selector, Wrapper_Selector, \
    Supervised_Filter, Unsupervised_Filter
from minder_utils.evaluate.evaluate_models import evaluate_features
from minder_utils.util.initial import first_run
from minder_utils.configurations import feature_selector_config
from minder_utils.models.classifiers.torch_classifiers import Classifiers
from minder_utils.dataloader import create_labelled_loader
from minder_utils.dataloader import Dataloader
from minder_utils.dataloader import process_data
from sklearn.linear_model import LogisticRegression
import os
import pandas as pd
import numpy as np

os.chdir('..')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# If you haven't processed the data, please uncomment the line below
# first_run()

labelled_data = Dataloader(None).labelled_data
unlabelled_data = Dataloader(None).unlabelled_data['activity']

# The script below can evaluate the importance of each feature

# ----------------- Intrinsic Selector ----------------- #
unlabelled, X, y, p_ids = process_data(labelled_data, unlabelled_data, num_days_extended=0)
X = np.mean(X.reshape(X.shape[0], 24, 14), axis=1)
y[y < 0] = 0
y = y.reshape(-1, )
train_dataloader, test_dataloader = create_labelled_loader(X, y, split=True)
model = Intrinsic_Selector(Classifiers('lr', 14, initial_manually=False),
                           **feature_selector_config['intrinsic_selector']['model'])
model.fit(train_dataloader)
model.test(test_dataloader)
importance = model.get_importance(train_dataloader, normalise=True)


# ----------------- Wrapper Selector ----------------- #
def run_select_features(X, y, p_ids, features, num_feats=1):
    selector = Wrapper_Selector(LogisticRegression(), model_name='rfecv', num_features=num_feats)
    selector.fit(X, y)
    # do the selection
    X = selector.transform(X)
    df = evaluate_features(X, y, p_ids)
    df['number_feature_select'] = num_feats
    df['selected features'] = ','.join(features[selector.mask_of_features()])
    return


# ----------------- Supervised Filter Selector ----------------- #
model = Supervised_Filter()
model.fit(X, y)
model.transform(X)

# ----------------- Unsupervised Filter Selector ----------------- #
model = Unsupervised_Filter()
model.fit(X)
model.transform(X)
