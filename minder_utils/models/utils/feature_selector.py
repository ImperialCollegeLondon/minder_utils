from abc import ABC, abstractmethod


class Feature_selector(ABC):
    def __init__(self, model):
        self.name = self.methods[model]
        self.model = getattr(self, model)()

    @property
    @abstractmethod
    def methods(self):
        pass

    def reset_model(self, model_name):
        self.name = self.methods[model_name]
        self.model = getattr(self, model_name)()

    def get_info(self, verbose=False):
        if verbose:
            print('Available methods:')
            for idx, key in enumerate(self.methods):
                print(str(idx).ljust(10, ' '), key.ljust(10, ' '), self.methods[key].ljust(10, ' '))
        return self.methods

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X):
        pass
