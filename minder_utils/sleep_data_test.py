from minder_utils.models.feature_extractors.keras_autoencoders import Extractor
from minder_utils.models.classifiers.classifiers import Classifiers
from minder_utils.scripts.weekly_loader import Weekly_dataloader
from sklearn.model_selection import train_test_split
from minder_utils.formatting.format_util import y_to_categorical
from minder_utils.evaluate.eval_utils import get_scores
from minder_utils.models.utils.util import train_test_scale
from minder_utils.formatting.format_util import format_mean_std
from sklearn.preprocessing import StandardScaler
from minder_utils.formatting.format_util import normalise
from minder_utils.configurations import config
import pandas as pd
import numpy as np
import os

loader = Weekly_dataloader(num_days_extended=1)


# Define a model for evaluation

class Semi_Classifiers:
    def __init__(self, ext_type, clf_type):
        assert ext_type in ['all', 'env', 'activity', 'phy']
        self.ext_type = ext_type
        self.clf_type = clf_type
        self.classifier = Classifiers(self.clf_type, boosting=False)

        # Train extractors
        unlabelled = np.load(os.path.join(loader.previous_unlabelled_data, 'activity.npy')).reshape(-1, 3 * 8 * 14)
        unlabelled = normalise(unlabelled.reshape(unlabelled.shape[0], 24, -1)).reshape(unlabelled.shape)
        self.act_extractor = Extractor(save_path='./data/weekly_test/model')
        self.act_extractor.train(unlabelled, 'cnn')

        unlabelled = np.load(os.path.join(loader.previous_unlabelled_data, 'environmental.npy'))
        unlabelled = StandardScaler().fit_transform(unlabelled)
        self.env_extractor = Extractor(save_path='./data/weekly_test/environmental')
        self.env_extractor.train(unlabelled, 'nn', input_dim=20, encoding_dim=5)

        unlabelled = np.load(os.path.join(loader.previous_unlabelled_data, 'physiological.npy'))
        unlabelled = StandardScaler().fit_transform(unlabelled)
        self.phy_extractor = Extractor(save_path='./data/weekly_test/physiological')
        self.phy_extractor.train(unlabelled, 'nn', input_dim=14, encoding_dim=5)

    def transform(self, X):
        if isinstance(X, list):
            act_data = self.act_extractor.transform(X[0], 'cnn')
            env_data = self.env_extractor.transform(X[1], 'nn')
            phy_data = self.phy_extractor.transform(X[2], 'nn')
        else:
            act_data = X[:, :-34].reshape(X.shape[0], 3, 8, 14)
            env_data = X[:, -34:-14]
            phy_data = X[:, -14:]
            act_data = self.act_extractor.transform(act_data, 'cnn')
            env_data = self.env_extractor.transform(env_data, 'nn')
            phy_data = self.phy_extractor.transform(phy_data, 'nn')
        if self.ext_type == 'all':
            return np.concatenate([act_data, env_data, phy_data], axis=1)
        elif self.ext_type == 'activity':
            return act_data
        elif self.ext_type == 'env':
            return env_data
        elif self.ext_type == 'phy':
            return phy_data
        else:
            raise ValueError('not a valid type')

    @property
    def model_type(self):
        return self.ext_type + '-' + self.clf_type

    def reset(self):
        self.classifier = Classifiers(self.clf_type, boosting=False)

    def predict_proba(self, X):
        X = self.transform(X)
        return self.classifier.predict_probs(X)

    def fit(self, X, y):
        X = self.transform(X)
        try:
            self.classifier.fit(X, y)
        except ZeroDivisionError:
            pass

    def predict(self, X):
        X = self.transform(X)
        return self.classifier.predict(X)


# Labelled data
X = np.load(os.path.join(loader.previous_labelled_data, 'activity.npy')).reshape(-1, 3, 8, 14)
env_X = np.load(os.path.join(loader.previous_labelled_data, 'environmental.npy'))
phy_X = np.load(os.path.join(loader.previous_labelled_data, 'physiological.npy'))
y = np.load(os.path.join(loader.previous_labelled_data, 'label.npy'), allow_pickle=True).astype(int)
p_ids = np.load(os.path.join(loader.previous_labelled_data, 'patient_id.npy'), allow_pickle=True)

X = normalise(X.reshape(X.shape[0], 24, -1)).reshape(X.shape)
env_X = StandardScaler().fit_transform(env_X)
phy_X = StandardScaler().fit_transform(phy_X)



def split_by_ids(X, y, patient_ids, cat=True, valid_only=True, stratify=True, seed=0):
    y[y == 0] = -1
    y = y.reshape(-1, )
    patient_ids = patient_ids.reshape(-1, )
    # make sure the train and test set got both positive and negative patients
    y_p_id = []
    for p_id in np.unique(patient_ids):
        _y = np.unique(y[patient_ids == p_id])
        rng = np.random.default_rng(seed)
        y_p_id.append(int(_y[0]) if len(_y) < 2 else rng.integers(0, 2))
        seed += 1
    y_p_id = np.array(y_p_id)
    y_p_id[y_p_id < 0] = 0

    train_ids, test_ids = train_test_split(np.unique(patient_ids), test_size=0.33, random_state=seed,
                                           stratify=y_p_id if stratify else None)
    test_y = y[np.isin(patient_ids, test_ids)]
    if valid_only:
        test_filter = np.isin(test_y, [-1, 1])
    else:
        test_filter = np.isin(test_y, np.unique(test_y))
    if cat:
        return [X[0][np.isin(patient_ids, train_ids)], X[1][np.isin(patient_ids, train_ids)],
                X[2][np.isin(patient_ids, train_ids)]], y_to_categorical(
            y[np.isin(patient_ids, train_ids)]), \
               [X[0][np.isin(patient_ids, test_ids)][test_filter],
                X[1][np.isin(patient_ids, test_ids)][test_filter],
                X[2][np.isin(patient_ids, test_ids)][test_filter]], y_to_categorical(y[np.isin(patient_ids, test_ids)][
                                                                                         test_filter])
    return [X[0][np.isin(patient_ids, train_ids)], X[1][np.isin(patient_ids, train_ids)],
            X[2][np.isin(patient_ids, train_ids)]], y[
        np.isin(patient_ids, train_ids)], \
           [X[0][np.isin(patient_ids, test_ids)], X[1][np.isin(patient_ids, test_ids)],
            X[2][np.isin(patient_ids, test_ids)]], y[
               np.isin(patient_ids, test_ids)]


def evaluate(model, X, y, p_ids, num_runs=10, valid_only=True, return_raw=False, scale_data=False):
    '''
    This function is used to evaluate the performance of your model
    Parameters
    ----------
    model
    X
    y
    p_ids
    num_runs
    valid_only
    return_raw

    Returns
    -------

    '''
    raw_results, results, sen, spe, accs, f1s = [], [], [], [], [], []
    header = ['model', 'sensitivity', 'specificity', 'acc', 'f1']
    for run in range(num_runs):
        X_train, y_train, X_test, y_test = split_by_ids(X, y, p_ids, seed=run, cat=valid_only, valid_only=valid_only)

        if scale_data:
            act_X_train, act_X_test = train_test_scale(X_train[0].reshape(-1, 3 * 8 * 14),
                                                       X_test[0].reshape(-1, 3 * 8 * 14))
            act_X_train, act_X_test = act_X_train.reshape(-1, 3, 8, 14), act_X_test.reshape(-1, 3, 8, 14)
            env_X_train, env_X_test = train_test_scale(X_train[1], X_test[1])
            phy_X_train, phy_X_test = train_test_scale(X_train[2], X_test[2])
            X_train, X_test = [act_X_train, env_X_train, phy_X_train], [act_X_test, env_X_test, phy_X_test]

        model.reset()
        model.fit(X_train, y_train)
        sensitivity, specificity, acc, f1 = get_scores(y_test, model.predict(X_test))
        if sensitivity is not None and str(sensitivity) != 'nan':
            sen.append(sensitivity)
            spe.append(specificity)
            accs.append(acc)
            f1s.append(f1)
            if return_raw:
                raw_results.append([model.model_type, sensitivity, specificity, acc, f1])
    row = [model.model_type, format_mean_std(sen), format_mean_std(spe), format_mean_std(accs), format_mean_std(f1s)]
    results.append(row)

    if return_raw:
        return pd.DataFrame(raw_results, columns=header)
    df_results = pd.DataFrame(results, columns=header)
    return df_results


model = Semi_Classifiers('all', 'bayes')

model.ext_type = 'all'
model.clf_type = 'bayes'

# import shap
# import matplotlib.pyplot as plt
#
# X_train, y_train, X_test, y_test = split_by_ids([X, env_X, phy_X], y, p_ids, seed=1234, cat=True, valid_only=True)
# X_train = np.concatenate([X_train[0].reshape(X_train[0].shape[0], -1), X_train[1], X_train[2]], axis=1)
# X_test = np.concatenate([X_test[0].reshape(X_test[0].shape[0], -1), X_test[1], X_test[2]], axis=1)
# model.fit(X_train, y_train)
# explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
# shap_values = explainer.shap_values(X_test[:10], nsamples=20)
# shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], matplotlib=True)

results = []
for clf_type in Classifiers().methods.keys():
    for ext_type in ['all', 'env', 'activity', 'phy']:
        model.ext_type = ext_type
        model.clf_type = clf_type
        results.append(evaluate(model, [X, env_X, phy_X], y, p_ids, scale_data=True))

pd.concat(results).to_csv('results.csv')
