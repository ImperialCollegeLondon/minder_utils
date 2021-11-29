from minder_utils.models.utils import EarlyStopping
import torch.nn as nn
import torch


class Classifiers:
    '''
    This class contains multiple classifiers based on pytorch
    '''
    def __init__(self, model_type, num_features, initial_manually=False, num_outputs=2):
        '''
        Initialise the classifier
        Parameters
        ----------
        model_type: str, 'lr' or 'nn'
        num_features: int, input dim
        initial_manually: bool, initial the weights to ones
        num_outputs: int, output dim, default = 2
        '''
        self.model_type = model_type
        self.early_stop = EarlyStopping()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.model = getattr(self, model_type)()
        self.initial_manually = initial_manually
        if self.initial_manually:
            for param in self.model.parameters():
                param.data = nn.parameter.Parameter(torch.ones_like(param))

    def reset(self):
        self.model = getattr(self, self.model_type)()
        if self.initial_manually:
            for param in self.model.parameters():
                param.data = nn.parameter.Parameter(torch.ones_like(param))

    def nn(self):
        return nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_outputs)
        )

    def lr(self):
        return nn.Linear(self.num_features, self.num_outputs)

    def parameters(self):
        return self.model.parameters()

    def __call__(self, X):
        return self.model(X)

