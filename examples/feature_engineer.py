from minder_utils.models.feature_selectors import Intrinsic_Selector, Wrapper_Selector, \
    Supervised_Filter, Unsupervised_Filter
from minder_utils.configurations import config
from minder_utils.evaluate import evaluate_features
from minder_utils.formatting.label import label_by_week
from minder_utils.feature_engineering import Feature_engineer
from minder_utils.formatting import Formatting
from minder_utils.visualisation import Visual_Evaluation
import os
import pandas as pd
import numpy as np

os.chdir('..')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
visual = Visual_Evaluation()

fe = Feature_engineer(Formatting())
data = label_by_week(fe.activity)


def evaluate_data(input_data, sensors, name):
    input_data = label_by_week(input_data)
    raw_data = input_data[~input_data.valid.isna()]
    X = raw_data[sensors].to_numpy()
    y = raw_data.valid.to_numpy()
    p_ids = raw_data.id.to_numpy()
    df = evaluate_features(X, y, p_ids, return_raw=True)
    df['feature_type'] = name

    model = Supervised_Filter('f_class')
    model.fit(X, y)

    return df, model.get_importance()


results, importance = [], []
# ----------------- Raw data ----------------- #
res = evaluate_data(fe.raw_activity, config['activity']['sensors'], 'weekly_raw_data')
results.append(res[0])
importance.append(res[1])

# ----------------- Feature Engineering ----------------- #
res = evaluate_data(fe.activity, fe.info.keys(), 'Feature engineer')
results.append(res[0])
importance.append(res[1])

results = pd.concat(results)

print(results.groupby(['model', 'feature_type']).mean())

df = pd.DataFrame([np.concatenate([config['activity']['sensors'], list(fe.info.keys())]), np.concatenate(importance)]).transpose()
df.columns = ['sensors', 'importance']

df = pd.DataFrame(list(fe.info.keys()), importance[1]).reset_index()
df.columns = ['importance', 'sensors']
visual.reset(results, df)
visual.boxplot()
visual.importance_bar()
