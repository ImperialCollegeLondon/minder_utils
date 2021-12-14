from minder_utils.models.feature_selectors import Intrinsic_Selector, Wrapper_Selector, \
    Supervised_Filter, Unsupervised_Filter
from minder_utils.configurations import config
from minder_utils.evaluate import evaluate_features
from minder_utils.models.classifiers.classifiers import Classifiers
from minder_utils.formatting.label import label_by_week
from minder_utils.feature_engineering import Feature_engineer
from minder_utils.formatting import Formatting
from minder_utils.visualisation import Visual_Evaluation
import os
import pandas as pd
import numpy as np
import datetime

os.chdir('..')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
visual = Visual_Evaluation()

test_data = pd.read_csv('./minder_utils/data/weekly_test/fe.csv')
test_data = test_data[pd.to_datetime(test_data.time) > str(datetime.date.today() - datetime.timedelta(days=10))]

fe = Feature_engineer(Formatting())
data = label_by_week(fe.activity)

input_data = label_by_week(fe.activity)
raw_data = input_data[~input_data.valid.isna()]
X = raw_data[fe.info.keys()].to_numpy()
y = raw_data.valid.to_numpy()
y[y < 0] = 0
p_ids = raw_data.id.to_numpy()
test_x = test_data[fe.info.keys()].to_numpy()

ids = []
probabilities = []
for model_type in Classifiers().get_info():
    model = Classifiers(model_type)
    model.fit(X, y.astype(float))
    prediction = model.predict_probs(test_x)
    probabilities.append(list(model.predict_probs(test_x)[:, 1]))

probabilities = np.mean(np.array(probabilities), axis=0)
results = {'id': test_data.id.to_list(), 'prediction': probabilities > 0.5, 'confidence': probabilities}
results = pd.DataFrame.from_dict(results)

mask = results['prediction'] == False
results.confidence[mask] = 1 - results.confidence[mask]
