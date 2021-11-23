from sklearn.feature_selection import RFE, RFECV
from minder_utils.models.utils import Feature_selector
import numpy as np


class Supervised_wrapper(Feature_selector):
    '''
    This class provide a set of supervised feature selection methods.
    Particularly, it contains a set of filter methods, which will perform SEPARATELY with the classifier.

    Currently, it contains:
        - REF: Recursive feature elimination

    ```Example```
    ```
    from minder_utils.models.feature_selectors.supervised.wrapper import Supervised_wrapper
    from sklearn.svm import SVC

    selector = Supervised_wrapper(SVC(kernel='linear'), model_name='rfe')
    # show the available methods:
    selector.get_info(verbose=True)

    # train the selector
    selector.fit(X, y)

    # do the selection
    X = selector.transform(X)
    ```
    '''

    def __init__(self, estimator, model_name='rfe', num_features=10):
        '''
        Select a proportion of features
        Args:
            num_features: int / float, number / percentage of features to be selected

        '''
        self.estimator = estimator
        self.num_features = num_features
        super().__init__(model_name)

    def reset_model(self, model_name, num_features=None):
        self.num_features = self.num_features if num_features is None else num_features
        self.name = self.methods[model_name]
        self.model = getattr(self, model_name)()

    @property
    def methods(self):
        return {
            'rfe': 'Recursive feature elimination',
            'rfecv': 'Recursive feature elimination with cross-validation ',
        }

    def rfe(self):
        return RFE(self.estimator, n_features_to_select=self.num_features)

    def rfecv(self):
        return RFECV(self.estimator, min_features_to_select=self.num_features, cv=5)

    def fit(self, X, y):
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        return self.model.fit(X, y)

    def transform(self, X):
        return self.model.transform(X)

    def mask_of_features(self):
        return self.model.support_

    def __name__(self):
        return 'Supervised Filter', self.name
