from minder_utils.models.utils import Feature_selector
from minder_utils.models.utils import EarlyStopping
import torch.nn.functional as F
import torch
from torch import optim
from torch import nn
import numpy as np


class Intrinsic_Selector(Feature_selector):
    '''
    This class provide a set of supervised feature selection methods.
    Particularly, it contains a set of intrinsic methods, which will perform automatic feature selection
     DURING TRAINING.

    Currently, it contains:
        - Linear feature selector
    ```Example```
    ```
    ```
    '''

    def __init__(self, classifier, model_name, num_features, freeze_classifier=False, temperature=5):
        self.classifier = classifier
        self.num_features = num_features
        super().__init__(model_name)
        self.name = self.methods[model_name]
        self.early_stop = EarlyStopping(**self.config['early_stop'])
        self.freeze_classifier = freeze_classifier
        self.discrete = 'discrete' in model_name
        self.temperature = temperature

    def reset_model(self, model_name, discrete=True):
        self.discrete = discrete
        self.name = self.methods[model_name]
        self.model = getattr(self, model_name)()

    @property
    def methods(self):
        return {
            'linear': 'linear feature selector',
            'discrete_linear': 'discrete linear feature selector'
        }

    def linear(self):
        return nn.Linear(self.num_features, self.num_features, bias=False)

    def discrete_linear(self):
        return nn.ModuleList([
            nn.Linear(self.num_features, self.num_features, bias=False),
            nn.Linear(self.num_features, self.num_features, bias=False)])

    def fit(self, dataloader, num_epoch=50):
        parameters = self.model.parameters() if self.freeze_classifier \
            else list(self.model.parameters()) + list(self.classifier.parameters())
        optimiser = optim.Adam(parameters, lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for e in range(num_epoch):
            if self.early_stop.early_stop:
                break
            for X, y in dataloader:
                optimiser.zero_grad()
                if self.discrete:
                    features_importance = torch.stack([self.model[0](X), self.model[1](X)], dim=-1)
                    features_importance = F.gumbel_softmax(features_importance, tau=self.temperature, hard=True, dim=-1)[:, :, 1]
                    features_importance = features_importance
                else:
                    features_importance = self.model(X)
                    features_importance = F.softmax(features_importance, dim=1)
                X = X * features_importance
                outputs = self.classifier(X)
                loss = criterion(outputs, y) + torch.sum(features_importance)
                loss.backward()
                optimiser.step()
                print('Epoch: %d / %5d,  Loss: %.3f' %
                      (e + 1, num_epoch, loss.item()), end='\n')
                self.early_stop(loss.item(), self.model, self.__class__.__name__)
                if self.early_stop.early_stop and self.config['early_stop']['enable']:
                    break
            if self.early_stop.early_stop and self.config['early_stop']['enable']:
                break
            return self

    def test(self, dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in dataloader:
                if self.discrete:
                    features_importance = torch.stack([self.model[0](X), self.model[1](X)], dim=-1)
                    features_importance = F.softmax(features_importance / self.temperature, dim=-1)[:, :, 1]
                else:
                    features_importance = self.model(X)
                    features_importance = F.softmax(features_importance, dim=1)
                X *= features_importance
                outputs = self.classifier(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        print('Accuracy: %d %%' % (100 * correct / total))
        return 100 * correct / total

    def transform(self, X):
        pass

    def __name__(self):
        return 'Supervised Intrinsic Selector', self.name

    def get_importance(self, dataloader, normalise=True):
        importance = []
        with torch.no_grad():
            for X, y in dataloader:
                if self.discrete:
                    features_importance = torch.stack([self.model[0](X), self.model[1](X)], dim=-1)
                    importance.extend(list(F.softmax(features_importance / self.temperature, dim=-1).detach().numpy()[:, :, 1]))
                else:
                    features_importance = self.model(X)
                    importance.extend(list(F.softmax(features_importance, dim=1).detach().numpy()))
        importance = np.array(importance)
        if normalise:
            importance -= np.min(importance, axis=1, keepdims=True) - 1e-5
            importance /= np.max(importance, axis=1, keepdims=True)
        importance = np.mean(importance, axis=0)
        return importance
