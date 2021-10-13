from .models.feature_extractors.autoencoders import Extractor
from .models.classifiers.classifiers import Classifiers
import numpy as np
import pandas as pd
from .scirpts.weekly_loader import Weekly_dataloader
import os
from .formatting.map_utils import map_raw_ids
from .evaluate.evaluate_models import evaluate
from .formatting.format_util import y_to_categorical


def run_default(reload_weekly=False, reload_all=False):
    '''
    An example function for loading data and evaluating methods
    
    Arguments

        reload_weekly: bool: Download the previous week's data.
        
        reload_all: bool: Download all data.
    '''
    loader = Weekly_dataloader(num_days_extended=5)
    loader.load_data(reload_weekly, reload_all)
    unlabelled = np.load(os.path.join(loader.previous_data, 'unlabelled.npy'))
    X = np.load(os.path.join(loader.previous_data, 'X.npy'))
    y = np.load(os.path.join(loader.previous_data, 'y.npy'))
    label_p_ids = np.load(os.path.join(loader.previous_data, 'label_ids.npy'))

    extractor = Extractor()
    extractor.train(unlabelled, 'cnn')
    # Evaluate models
    evaluate(Classifiers('knn'), extractor.transform(X, 'cnn'), y, label_p_ids, 10)

    weekly_data = np.load(os.path.join(loader.weekly_data, 'unlabelled.npy'))
    p_ids = np.load(os.path.join(loader.weekly_data, 'patient_id.npy'))
    dates = np.load(os.path.join(loader.weekly_data, 'dates.npy'), allow_pickle=True)
    X = extractor.transform(X, 'cnn')
    weekly_data = extractor.transform(weekly_data, 'cnn')

    y = np.argmax(y_to_categorical(y), axis=1)
    clf = Classifiers('knn')
    clf.fit(X, y)
    print(clf.predict(weekly_data))

    prediction = clf.predict(weekly_data)
    probability = clf.predict_probs(weekly_data)
    df = {'patient id': p_ids, 'Date': dates, 'prediction': prediction,
          'confidence': probability[np.arange(probability.shape[0]), prediction]}
    df = pd.DataFrame(df)
    df['TIHM ids'] = map_raw_ids(df['patient id'], True)
    df.to_csv('./results/weekly_test/alerts.csv')
    
    return


def load_data_default(reload_weekly=False, reload_all=False):
    '''
    An example function that downloads and processes the data.
    
    Arguments:

        reload_weekly: bool: Download the previous week's data.
    
        reload_all: bool: Download all data.
    '''

    loader = Weekly_dataloader(num_days_extended=5)
    loader.load_data(reload_weekly, reload_all)
    unlabelled = np.load(os.path.join(loader.previous_data, 'unlabelled.npy'))
    X = np.load(os.path.join(loader.previous_data, 'X.npy'))
    y = np.load(os.path.join(loader.previous_data, 'y.npy'))
    label_p_ids = np.load(os.path.join(loader.previous_data, 'label_ids.npy'))

    return




if __name__ == '__main__':
    run_default()