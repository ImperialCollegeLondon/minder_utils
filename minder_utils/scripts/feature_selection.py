from minder_utils.models.feature_selectors.supervised.filter import Supervised_Filter
from minder_utils.models.feature_selectors.unsupervised.filter import Unsupervised_Filter
from prettytable import PrettyTable
from minder_utils.evaluate.eval_utils import split_by_ids, get_scores
from minder_utils.formatting.format_util import format_mean_std
import pandas as pd


class Feature_Selection:
    '''
    This class can help you to choose the best feature selection method for your model.

    To run the script, your model must implement ```reset()``` method, which is to
    reset the parameters of your model for testing.

    ```Example```
    ```
    from minder_utils.scripts.feature_selection import Feature_Selection
    from minder_utils.models.classifiers import Classifiers

    classifier = Classifiers('nn')
    feature_select = Feature_Selection(classifier)

    feature_select.evaluate(X, y, p_ids, num_runs=10)
    ```
    '''
    def __init__(self, model, proportion=90):
        self.model = model
        self.feature_selector = [Supervised_Filter(proportion=proportion), Unsupervised_Filter()]

    def evaluate(self, X, y, p_ids, num_runs=10):
        header = ['Supervision', 'Method', 'Feature selected ({} in total)'.format(X.shape[1]), 'sensitivity', 'specificity', 'acc', 'f1']
        results = []
        for selector in self.feature_selector:
            sen, spe, accs, f1s, num_feats = [], [], [], [], []
            info = selector.get_info()
            for model_type in info:
                for run in range(num_runs):
                    X_train, y_train, X_test, y_test = split_by_ids(X, y, p_ids, seed=run)
                    self.model.reset()
                    selector.reset_model(model_type)
                    selector.fit(X_train, y_train)
                    X_train_trans = selector.transform(X_train)
                    X_test_trans = selector.transform(X_test)
                    self.model.fit(X_train_trans, y_train)
                    sensitivity, specificity, acc, f1 = get_scores(y_test, self.model.predict(X_test_trans))
                    if sensitivity is not None and str(sensitivity) != 'nan':
                        sen.append(sensitivity)
                        spe.append(specificity)
                        accs.append(acc)
                        f1s.append(f1)
                        num_feats.append(X_train_trans.shape[1])
                row = [selector.__name__()[0], selector.__name__()[1], format_mean_std(num_feats),
                       format_mean_std(sen), format_mean_std(spe), format_mean_std(accs), format_mean_std(f1s)]
                results.append(row)
        df_results = pd.DataFrame(results, columns=header)
        return df_results



