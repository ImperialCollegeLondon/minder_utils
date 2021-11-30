from minder_utils.evaluate.eval_utils import split_by_ids, get_scores
from minder_utils.formatting.format_util import format_mean_std
from minder_utils.models.classifiers.classifiers import Classifiers as keras_clf
import pandas as pd


def evaluate(model, X, y, p_ids, num_runs=10, valid_only=True, return_raw=False):
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


def evaluate_features(X, y, p_ids, num_runs=10, valid_only=True, return_raw=False):
    '''
    This function is to evaluate your features on the baseline models
    Parameters
    ----------
    X
    y
    p_ids
    num_runs
    valid_only
    return_raw

    Returns Dataframe, contains the performance of the models
    -------

    '''
    results = []
    for model_type in keras_clf().get_info():
        print('Evaluating ', model_type)
        clf = keras_clf(model_type)
        results.append(evaluate(clf, X, y, p_ids, valid_only=valid_only, num_runs=num_runs, return_raw=return_raw))
    return pd.concat(results)
