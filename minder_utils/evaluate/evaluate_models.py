from minder_utils.evaluate.eval_utils import split_by_ids, get_scores
from minder_utils.formatting.format_util import format_mean_std
import pandas as pd


def evaluate(model, X, y, p_ids, num_runs=10, valid_only=True):
    results, sen, spe, accs, f1s = [], [], [], [], []
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
    row = [model.model_type, format_mean_std(sen), format_mean_std(spe), format_mean_std(accs), format_mean_std(f1s)]
    results.append(row)

    df_results = pd.DataFrame(results, columns=header)
    return df_results
