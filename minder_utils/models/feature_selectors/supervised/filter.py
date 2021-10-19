from sklearn.feature_selection import SelectPercentile, chi2, f_classif, mutual_info_classif
from minder_utils.models.feature_selectors import Feature_selector_template
import numpy as np


class Supervised_Filter(Feature_selector_template):
    '''
    This class provide a set of supervised feature selection methods.
    Particularly, it contains a set of filter methods, which will perform separately with the classifier.

    Currently, it contains:
        - chi-squared stats
        - ANOVA F-value
        - mutual information

    ```Example```
    ```
    from minder_utils.models.feature_selectors.unsupervised.filter import Unsupervised_Filter

    selector = Unsupervised_Filter(model='vt')
    # show the available methods:
    selector.get_info(verbose=True)

    # train the selector. Note the X is the data, y is None and will not be used
    selector.fit(X, y)

    # do the selection
    X = selector.transform(X)
    ```
    '''
    def __init__(self, model='chi', proportion=90):
        '''
        Select a proportion of features
        Args:
            model: method to calculate the score for feature selection
            proportion: percentage of features to keep
        '''
        super().__init__(model)
        self.selector = SelectPercentile(self.model, percentile=proportion)
        self.proportion = proportion

    def reset_model(self, model, proportion=None):
        proportion = self.proportion if proportion is None else proportion
        self.name = self.methods[model]
        self.model = getattr(self, model)()
        self.selector = SelectPercentile(self.model, percentile=proportion)

    @property
    def methods(self):
        return {
            'chi': 'chi-squared stats',
            'f_class': 'ANOVA F-value',
            'mi': 'mutual information',
        }

    @staticmethod
    def chi():
        return chi2

    @staticmethod
    def f_class():
        return f_classif

    @staticmethod
    def mi():
        return mutual_info_classif

    def fit(self, X, y):
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        return self.selector.fit(X, y)

    def transform(self, X):
        return self.selector.transform(X)

    def __name__(self):
        return 'Supervised Filter', self.name

