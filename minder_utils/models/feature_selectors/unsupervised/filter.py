from sklearn.feature_selection import VarianceThreshold
from minder_utils.models.feature_selectors.template import Feature_selector_template


class Unsupervised_Filter(Feature_selector_template):
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
