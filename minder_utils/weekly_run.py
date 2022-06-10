from minder_utils.models.feature_extractors.keras_autoencoders import Extractor
from minder_utils.models.classifiers.classifiers import Classifiers
import numpy as np
import pandas as pd
from minder_utils.scripts.weekly_loader import Weekly_dataloader
import os
from minder_utils.formatting.map_utils import map_raw_ids
from minder_utils.evaluate.evaluate_models import evaluate
from minder_utils.formatting.format_util import y_to_categorical
from minder_utils.formatting.format_util import normalise
import ruamel.yaml



class Weekly_alerts:
    def __init__(self, autoencoder='cnn', normalisation=True):
        self.normalisation = normalisation
        self.loader = Weekly_dataloader(num_days_extended=1)
        self.loader.refresh()
        self.reset()

        unlabelled = np.load(os.path.join(self.loader.previous_unlabelled_data, 'activity.npy'))
        unlabelled = unlabelled.reshape(-1, 3, 8, 14)
        unlabelled = normalise(unlabelled.reshape(unlabelled.shape[0], 24, -1)).reshape(unlabelled.shape)
        extractor = Extractor(save_path='./data/weekly_test/model')
        extractor.train(unlabelled, autoencoder, normalisation='l2')

        #sleep_extractor = Extractor(save_path='./data/weekly_test/sleep_model')
        #unlabelled = np.load(os.path.join(self.loader.previous_unlabelled_data, 'sleep.npy'))
        #sleep_extractor.train(unlabelled, 'nn')

        self.extractor = extractor
        #self.sleep_extractor = sleep_extractor
        self.autoencoder = autoencoder

    def reset(self, num_days_extended=0):
        X = np.load(os.path.join(self.loader.previous_labelled_data, 'activity.npy'))
        y = np.load(os.path.join(self.loader.previous_labelled_data, 'label.npy'), allow_pickle=True).astype(int)
        label_p_ids = np.load(os.path.join(self.loader.previous_labelled_data, 'patient_id.npy'), allow_pickle=True).astype(str)

        X = X.reshape(-1, 3, 8, 14)

        # labelled data
        #indices = list(y[0][1: num_days_extended * 2 + 1]) + [-1, 1]
        #X = X[np.isin(y, indices)].reshape(-1, 3, 8, 14)
        #label_p_ids = label_p_ids[np.isin(y, indices)]
        #y = y[np.isin(y, indices)]
        #y[y > 0] = 1
        #y[y < 0] = -1

        # test data
        weekly_data = np.load(os.path.join(self.loader.current_data, 'activity.npy'))
        p_ids = np.load(os.path.join(self.loader.current_data, 'patient_id.npy'), allow_pickle=True)
        dates = np.load(os.path.join(self.loader.current_data, 'dates.npy'), allow_pickle=True)
        weekly_data = weekly_data.reshape(-1, 3, 8, 14)

        if self.normalisation:
            X = normalise(X.reshape(X.shape[0], 24, -1)).reshape(X.shape)
            weekly_data = normalise(weekly_data.reshape(weekly_data.shape[0], 24, -1)).reshape(weekly_data.shape)
        self.data = {
            'labelled': (X, y, label_p_ids),
            'test': (weekly_data, p_ids, dates)
        }

    def evaluate(self, transform=True, boosting=True):
        df = []
        for clf in Classifiers().methods:
            df.append(self._evaluate(clf, transform, boosting))
        print(pd.concat(df))

    def _evaluate(self, clf_type, transform=True, boosting=True):
        X, y, label_p_ids = self.data['labelled']
        if transform:
            X = self.extractor.transform(X, self.autoencoder, normalisation='l2')
        else:
            X = X.reshape(X.shape[0], -1)
        return evaluate(Classifiers(clf_type, boosting), X, y, label_p_ids, 10)

    def predict(self, transform=True, boosting=True):
        weekly_data, p_ids, dates = self.data['test']

        probability = []
        for clf in Classifiers().methods:
            prob = self._predict(clf, transform=transform, boosting=boosting)
            probability.append(prob)

        probability = np.mean(probability, axis=0)
        prediction = np.argmax(probability, axis=1)
        df = {'patient id': p_ids, 'Date': dates, 'prediction': prediction,
              'confidence': probability[np.arange(probability.shape[0]), prediction]}
        df['TIHM ids'] = df['patient id']#map_raw_ids(df['patient id'], True)
        df = pd.DataFrame(df)
        df.to_csv('./results/weekly_test/alerts.csv')
        return df

    def _predict(self, clf_type, return_df=False, transform=True, boosting=True):
        weekly_data, p_ids, dates = self.data['test']
        X, y, label_p_ids = self.data['labelled']
        if transform:
            X = self.extractor.transform(X, self.autoencoder, normalisation='l2')
            weekly_data = self.extractor.transform(weekly_data, self.autoencoder, normalisation='l2')
        else:
            X = X.reshape(X.shape[0], -1)
            weekly_data = weekly_data.reshape(weekly_data.shape[0], -1)
        y[y < 0] = 0
        y = y_to_categorical(y) if clf_type in ['nn', 'lstm'] else np.argmax(y_to_categorical(y), axis=1)
        clf = Classifiers(clf_type, boosting)
        clf.fit(X, y)
        probability = clf.predict_probs(weekly_data)
        if return_df:
            prediction = np.argmax(probability, axis=1)
            df = {'patient id': p_ids, 'Date': dates, 'prediction': prediction,
                  'confidence': probability[np.arange(probability.shape[0]), prediction]}
            df['TIHM ids'] = map_raw_ids(df['patient id'], True)
            df = pd.DataFrame(df)
            df.to_csv('./results/weekly_test/alerts.csv')
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
    df = df[df['patient id'] == p_id].copy()
    if by_days and df.prediction.sum() < 3:
        return 0
    df.prediction = df.prediction.map({0: 1, 1: 0})
    df.confidence = np.abs(df.confidence - df.prediction)
    return df.confidence.mean()


def feature_engineering():
    from minder_utils.feature_engineering import Feature_engineer
    from minder_utils.formatting import Formatting
    fe = Feature_engineer(Formatting('./data/weekly_test/current/csv'))
    return fe.activity


def get_results(model, transform, boosting):
    df = model.predict(transform=transform, boosting=boosting)
    for p_id in df['patient id'].unique():
        value = cal_confidence(df, p_id)
        if value > 0.5:
            print(p_id, value, df[df['patient id'] == p_id]['TIHM ids'].unique())


if __name__ == '__main__':
    # data = feature_engineering()
    wa = Weekly_alerts()
    df = wa.predict(True, False)
    for p_id in df['patient id'].unique():
        res = cal_confidence(df, p_id, True)
        if res > 0.5:
            print(p_id, res, df[df['patient id'] == p_id]['TIHM ids'].unique())
