from sklearn.feature_selection import VarianceThreshold
from minder_utils.models.feature_selectors import Feature_selector_template


class Unsupervised_Filter(Feature_selector_template):
    '''
    This class provide a set of unsupervised feature selection methods.

    Currently, it contains:
        - VarianceThreshold

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
    def __init__(self, model='vt'):
        super().__init__(model)

    @property
    def methods(self):
        return {
            'vt': 'VarianceThreshold',
        }

    @staticmethod
    def vt():
        return VarianceThreshold()

    def __name__(self):
        return 'Unsupervised Filter', self.name

    def fit(self, X, y=None):
        return self.model.fit(X)

    def transform(self, X):
        return self.model.transform(X)