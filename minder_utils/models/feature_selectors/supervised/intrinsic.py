from minder_utils.models.feature_selectors import Feature_selector_template
from minder_utils.models.utils import EarlyStopping
import torch.nn.functional as F
import torch
from torch import optim
from torch import nn
import numpy as np


class Intrinsic_selector(Feature_selector_template):
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

    def __init__(self, classifier, model_name, num_features, freeze_classifier=False, discrete=True):
        self.classifier = classifier
        self.num_features = num_features
        super().__init__(model_name)
        self.discrete = discrete
        self.name = 'discrete_' + self.methods[model_name] if self.discrete else self.methods[model_name]
        self.early_stop = EarlyStopping()
        self.freeze_classifier = freeze_classifier

    def reset_model(self, model_name, discrete=True):
        self.discrete = discrete
        self.name = 'discrete_' + self.methods[model_name] if self.discrete else self.methods[model_name]
        self.model = getattr(self, model_name)()

    @property
    def methods(self):
        return {
            'linear': 'linear feature selector',
        }

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
                features_importance = self.model(X)
                if self.discrete:
                    features_importance = F.gumbel_softmax(features_importance, hard=True)
                features_importance = torch.mean(features_importance, dim=0)
                X = X * features_importance
                outputs = self.classifier(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimiser.step()
                print('Epoch: %d / %5d,  Loss: %.3f' %
                      (e + 1, num_epoch, loss.item()), end='\n')
                self.early_stop(loss.item(), self.model)
                if self.early_stop.early_stop:
                    break
            print('')

    def test(self, dataloader, T=1e-5):
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in dataloader:
                features_importance = self.model(X)
                if self.discrete:
                    features_importance = F.softmax(features_importance / T)
                X *= features_importance
                outputs = self.classifier(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == np.argmax(y, axis=1)).sum().item()

        print('Accuracy: %d %%' % (100 * correct / total))
        return 100 * correct / total

    def transform(self, X):
        pass

    def linear(self):
        return nn.Linear(self.num_features, self.num_features, bias=False)

    def __name__(self):
        return 'Supervised Intrinsic Selector', self.name

    def get_importance(self, dataloader, datatype):
        importance = []
        with torch.no_grad():
            for X, y in dataloader:
                features_importance = self.model(X)
                if self.discrete:
                    importance.extend(list(F.softmax(features_importance / 1e-5, dim=1).detach().numpy()))
                else:
                    importance.extend(list(F.softmax(features_importance, dim=1).detach().numpy()))
        importance = np.array(importance)
        if datatype == 'activity':
            importance = importance.reshape(importance.shape[0], 24, -1)
            importance = np.sum(importance, axis=1)
        return np.mean(importance, axis=0)
