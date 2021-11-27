from minder_utils.models.feature_extractors.keras_autoencoders import Extractor
from minder_utils.models.classifiers.classifiers import Classifiers
import numpy as np
import pandas as pd
from minder_utils.scripts.weekly_loader import Weekly_dataloader
import os
from minder_utils.formatting.map_utils import map_raw_ids
from minder_utils.evaluate.evaluate_models import evaluate
from minder_utils.formatting.format_util import y_to_categorical

pd.set_option('max_columns', None)
pd.set_option('max_colwidth', None)


class Weekly_alerts:
    def __init__(self, autoencoder='cnn'):
        self.loader = Weekly_dataloader(num_days_extended=6)
        # check the collate next week
        self.loader.refresh()
        self.reset()

        unlabelled = np.load(os.path.join(self.loader.previous_unlabelled_data, 'activity.npy'))
        unlabelled = unlabelled.reshape(-1, 3, 8, 14)
        extractor = Extractor()
        extractor.train(unlabelled, autoencoder)

        self.extractor = extractor
        self.autoencoder = autoencoder

    def reset(self, num_days_extended=0):
        X = np.load(os.path.join(self.loader.previous_labelled_data, 'activity.npy'))
        y = np.load(os.path.join(self.loader.previous_labelled_data, 'label.npy'))
        label_p_ids = np.load(os.path.join(self.loader.previous_labelled_data, 'patient_id.npy'))

        # labelled data
        indices = list(y[0][1: num_days_extended * 2 + 1]) + [-1, 1]
        X = X[np.isin(y, indices)].reshape(-1, 3, 8, 14)
        label_p_ids = label_p_ids[np.isin(y, indices)]
        y = y[np.isin(y, indices)]
        y[y > 0] = 1
        y[y < 0] = -1

        # test data
        weekly_data = np.load(os.path.join(self.loader.current_data, 'activity.npy'))
        p_ids = np.load(os.path.join(self.loader.current_data, 'patient_id.npy'))
        dates = np.load(os.path.join(self.loader.current_data, 'dates.npy'), allow_pickle=True)
        weekly_data = weekly_data.reshape(-1, 3, 8, 14)

        self.data = {
            'labelled': (X, y, label_p_ids),
            'test': (weekly_data, p_ids, dates)
        }

    def evaluate(self):
        df = []
        for clf in Classifiers().methods:
            df.append(self._evaluate(clf))
        print(pd.concat(df))

    def _evaluate(self, clf_type):
        X, y, label_p_ids = self.data['labelled']
        return evaluate(Classifiers(clf_type), self.extractor.transform(X, self.autoencoder), y, label_p_ids, 10)

    def predict(self):
        weekly_data, p_ids, dates = self.data['test']

        probability = []
        for clf in Classifiers().methods:
            prob = self._predict(clf)
            probability.append(prob)

        probability = np.mean(probability, axis=0)
        prediction = np.argmax(probability, axis=1)
        df = {'patient id': p_ids, 'Date': dates, 'prediction': prediction,
              'confidence': probability[np.arange(probability.shape[0]), prediction]}
        df['TIHM ids'] = map_raw_ids(df['patient id'], True)
        df = pd.DataFrame(df)
        df.to_csv('../results/weekly_test/alerts.csv')
        return df

    def _predict(self, clf_type, return_df=False):
        weekly_data, p_ids, dates = self.data['test']
        X, y, label_p_ids = self.data['labelled']
        y[y < 0] = 0
        y = y_to_categorical(y) if clf_type in ['nn', 'lstm'] else np.argmax(y_to_categorical(y), axis=1)
        clf = Classifiers(clf_type)
        clf.fit(self.extractor.transform(X, self.autoencoder), y)
        probability = clf.predict_probs(weekly_data)
        if return_df:
            prediction = np.argmax(probability, axis=1)
            df = {'patient id': p_ids, 'Date': dates, 'prediction': prediction,
                  'confidence': probability[np.arange(probability.shape[0]), prediction]}
            df['TIHM ids'] = map_raw_ids(df['patient id'], True)
            df = pd.DataFrame(df)
            df.to_csv('../results/weekly_test/alerts.csv')
            return df
        return probability


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


def cal_confidence(df, p_id, by_days=False):
    df = df[df['patient id'] == p_id]
    if by_days and df.prediction.sum() < 3:
        return 0
    df.prediction = df.prediction.map({0: 1, 1: 0})
    df.confidence = np.abs(df.confidence - df.prediction)
    return df.confidence.mean()


if __name__ == '__main__':
    wa = Weekly_alerts()
    wa.evaluate()
    df = wa._predict('bayes', return_df=True)
    for p_id in df['patient id'].unique():
        value = cal_confidence(df, p_id)
        if value > 0.5:
            print(p_id, value)
