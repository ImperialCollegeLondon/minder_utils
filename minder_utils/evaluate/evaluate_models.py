from minder_utils.evaluate.eval_utils import split_by_ids, get_scores
from minder_utils.formatting.format_util import format_mean_std
from minder_utils.models.classifiers.classifiers import Classifiers as keras_clf
from minder_utils.models.utils.util import train_test_scale
import pandas as pd
import numpy as np



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
            X_train, X_test = train_test_scale(X_train, X_test)

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


def evaluate_features(X, y, p_ids, num_runs=10, valid_only=True, return_raw=False, verbose = True, scale_data=False):
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
    for model_type in keras_clf(boosting=False).get_info():
        if verbose:
            print('Evaluating ', model_type)
        clf = keras_clf(model_type)
        results.append(evaluate(clf, X, y, p_ids, valid_only=valid_only, num_runs=num_runs, return_raw=return_raw, scale_data=scale_data))
    return pd.concat(results)





def evaluate_features_loo(X, y, p_ids, num_runs = 10, nice_names_X_columns = None, scale_data=True):
    '''
    This function makes use of the above two functions to calculate the relative metrics
    when one of the features is left out.


    '''

    results_all = evaluate_features(X=X, y=y, p_ids=p_ids, num_runs=num_runs, return_raw=True,  verbose = False, scale_data=scale_data)
    results_all_mean = results_all.groupby('model').mean()

    dividing_values = results_all_mean.to_dict('index')

    relative_result_list = []

    def relative_group_by(x):
        model_name = x['model'][0]
        divide_vector = np.asarray(list(dividing_values[model_name].values()))
        x = x[['sensitivity', 'specificity', 'acc', 'f1']]/divide_vector
        return x

    for col_index_out in range(X.shape[1]):
        
        X_to_test = np.delete(X, obj=col_index_out, axis=1)

        results_to_test = evaluate_features(X=X_to_test, y=y, p_ids=p_ids, num_runs=num_runs, return_raw=True, verbose = False)
        
        results_to_test['column_out'] = col_index_out if nice_names_X_columns is None else nice_names_X_columns[col_index_out]

        results_to_test[['sensitivity', 'specificity', 'acc', 'f1']] = results_to_test.groupby(by=['model']).apply(relative_group_by)


        relative_result_list.append(results_to_test)

    relative_result_all = pd.concat(relative_result_list)
    relative_result_all_melt = relative_result_all.melt(id_vars=['model', 'column_out'], var_name = 'metric', value_name='value')

    return relative_result_all_melt


