from .eval_utils import split_by_ids, get_scores
from ..formatting.format_util import format_mean_std
import pandas as pd


def evaluate(model, X, y, p_ids, num_runs=10):
    results, sen, spe, accs, f1s = [], [], [], [], []
    header = ['sensitivity', 'specificity', 'acc', 'f1']
    for run in range(num_runs):
        X_train, y_train, X_test, y_test = split_by_ids(X, y, p_ids, seed=run)
        model.re_initialise()
        model.fit(X_train, y_train)
        sensitivity, specificity, acc, f1 = get_scores(y_test, model.predict(X_test))
        if sensitivity is not None and str(sensitivity) != 'nan':
            sen.append(sensitivity)
            spe.append(specificity)
            accs.append(acc)
            f1s.append(f1)
    row = [format_mean_std(sen), format_mean_std(spe), format_mean_std(accs), format_mean_std(f1s)]
    results.append(row)

    df_results = pd.DataFrame(results, columns=header)
    print(df_results)
    return df_results
