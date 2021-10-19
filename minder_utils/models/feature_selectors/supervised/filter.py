from sklearn.feature_selection import SelectPercentile, chi2
from minder_utils.models.feature_selectors.template import Feature_selector_template


class Supervised_Filter(Feature_selector_template):
    def __init__(self, model='chi', proportion=90):
        super().__init__(model)
        self.selector = SelectPercentile(self.model, percentile=proportion)
        self.proportion = proportion

    def reset_model(self, model, proportion=None):
        proportion = self.proportion if proportion is None else proportion
        self.name = model
        self.model = getattr(self, model)()
        self.selector = SelectPercentile(self.model, percentile=proportion)

    @property
    def methods(self):
        return {
            'chi': 'chi-squared stats',
        }

    @staticmethod
    def chi():
        return chi2

    def fit(self, X, y):
        return self.selector.fit(X, y)

    def transform(self, X):
        return self.selector.transform(X)

    def __name__(self):
        return 'Supervised Filter', self.name

