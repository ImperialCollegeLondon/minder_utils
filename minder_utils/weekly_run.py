from minder_utils.models.feature_extractors.keras_autoencoders import Extractor
from minder_utils.models.classifiers.classifiers import Classifiers
import numpy as np
import pandas as pd
from minder_utils.scripts.weekly_loader import Weekly_dataloader
import os
from minder_utils.formatting.map_utils import map_raw_ids
from minder_utils.evaluate.evaluate_models import evaluate
from minder_utils.formatting.format_util import y_to_categorical


def run_default(clf_type='bayes'):
    '''
    An example function for loading data and evaluating methods on the activity data
    with UTIs as labels.
    
    Arguments
    ---------

    - reload_weekly: bool: 
        Download the previous week's data.
    
    - reload_all: bool: 
        Download all data.

    '''
    loader = Weekly_dataloader(num_days_extended=5)
    loader.refresh()
    unlabelled = np.load(os.path.join(loader.previous_unlabelled_data, 'activity.npy'))
    X = np.load(os.path.join(loader.previous_labelled_data, 'activity.npy'))
    y = np.load(os.path.join(loader.previous_labelled_data, 'label.npy'))
    label_p_ids = np.load(os.path.join(loader.previous_labelled_data, 'patient_id.npy'))

    unlabelled = unlabelled.reshape(-1, 3, 8, 14)
    X = X[np.isin(y, [-1, 1])].reshape(-1, 3, 8, 14)
    label_p_ids = label_p_ids[np.isin(y, [-1, 1])]
    y = y[np.isin(y, [-1, 1])]
    extractor = Extractor()
    extractor.train(unlabelled, 'cnn')
    # Evaluate models
    print(evaluate(Classifiers(clf_type), extractor.transform(X, 'cnn'), y, label_p_ids, 10))

    weekly_data = np.load(os.path.join(loader.current_data, 'activity.npy'))
    p_ids = np.load(os.path.join(loader.current_data, 'patient_id.npy'))
    dates = np.load(os.path.join(loader.current_data, 'dates.npy'), allow_pickle=True)

    weekly_data = weekly_data.reshape(-1, 3, 8, 14)

    X = extractor.transform(X, 'cnn')
    weekly_data = extractor.transform(weekly_data, 'cnn')

    y = np.argmax(y_to_categorical(y), axis=1)
    clf = Classifiers(clf_type)
    clf.fit(X, y)
    print(clf.predict(weekly_data))

    prediction = clf.predict(weekly_data)
    probability = clf.predict_probs(weekly_data)
    df = {'patient id': p_ids, 'Date': dates, 'prediction': prediction,
          'confidence': probability[np.arange(probability.shape[0]), prediction]}
    df = pd.DataFrame(df)
    df['TIHM ids'] = map_raw_ids(df['patient id'], True)
    df.to_csv('../results/weekly_test/alerts.csv')

    return df


def load_data_default(reload_weekly=False, reload_all=False):
    '''
    An example function that downloads and processes the activity data.
    
    Arguments:
    ---------

    - reload_weekly: bool: 
        Download the previous week's data.

    - reload_all: bool: 
        Download all data.
    
    '''

    loader = Weekly_dataloader(num_days_extended=5)
    loader.load_data(reload_weekly, reload_all)
    unlabelled = np.load(os.path.join(loader.previous_data, 'unlabelled.npy'))
    X = np.load(os.path.join(loader.previous_data, 'X.npy'))
    y = np.load(os.path.join(loader.previous_data, 'y.npy'))
    label_p_ids = np.load(os.path.join(loader.previous_data, 'label_ids.npy'))

    return


def cal_confidence(df, p_id):
    df = df[df['patient id'] == p_id]
    df.prediction = df.prediction.map({0: 1, 1: 0})
    df.confidence = np.abs(df.confidence - df.prediction)
    return df.confidence.mean()


if __name__ == '__main__':
    df = run_default('knn')
    for p_id in df['patient id'].unique():
        value = cal_confidence(df, p_id)
        if value > 0.5:
            print(p_id, value)
