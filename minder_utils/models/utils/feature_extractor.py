from abc import ABC, abstractmethod
from minder_utils.models.utils import EarlyStopping
from minder_utils.util import save_mkdir
import torch.nn as nn
import torch
import os
import numpy as np
from minder_utils.configurations import feature_extractor_config
from minder_utils.models.utils import get_device


class Feature_extractor(ABC, nn.Module):
    def __init__(self):
        super(Feature_extractor, self).__init__()
        self.early_stop = EarlyStopping(**self.config['early_stop'])
        self.device = get_device()

    @property
    def config(self) -> dict:
        return feature_extractor_config[self.__class__.__name__.lower()]

    @abstractmethod
    def step(self, data):
        pass

    def get_info(self, config=None, indent=0):
        if config is None:
            config = self.config
        for key, value in config.items():
            if isinstance(value, dict):
                print(' ' * indent + str(key))
                self.get_info(value, indent + 1)
            else:
                print(' ' * indent + str(key).ljust(10, ' '), str(value))

    def fit(self, train_loader, save_name=None):
        if save_name is None:
            save_name = self.__class__.__name__
        if not self.config['train']['retrain']:
            if self.load_pre_trained_weights(save_name):
                return

        self.model = self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), **self.config['optimiser'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)
        for epoch_counter in range(self.config['train']['epochs']):
            for data in train_loader:
                optimizer.zero_grad()
                loss = self.step(data).to(self.device)
                loss.backward()
                if self.config['train']['verbose']:
                    print('Epoch {}/{}, Loss: '.format(epoch_counter,
                                                       self.config['train']['epochs']), loss.item(), end='\n')
                optimizer.step()
                scheduler.step()
                self.early_stop(loss.item(), self.model, save_name)
                if self.early_stop.early_stop and self.config['early_stop']['enable']:
                    break
            if self.early_stop.early_stop and self.config['early_stop']['enable']:
                break

    def load_pre_trained_weights(self, save_name):
        try:
            checkpoints_folder = os.path.join(self.config['early_stop']['path'], save_name)
            state_dict = torch.load(checkpoints_folder)
            self.model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
            return True
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
            return False

    @staticmethod
    def which_data(data):
        return data

    def transform(self, test_loader):
        """
        :param test_loader: sample validated date only
        :return:
        """
        # validation steps
        with torch.no_grad():
            self.model.eval()
            features = []
            for data in test_loader:
                if not isinstance(data, torch.Tensor):
                    data = self.which_data(data)
                feat = self.model(data)
                if not isinstance(feat, torch.Tensor):
                    feat = feat[0]
                features.append(feat.numpy())

        if self.config['test']['save']:
            save_mkdir(self.config['test']['save_path'])
            np.save(os.path.join(self.config['test']['save_path'], self.__class__.__name__.lower() + '.npy'), np.concatenate(features))
            print('Test data has been transformed and saved to ',
                  os.path.join(self.config['test']['save_path'], self.__class__.__name__).lower() + '.npy')

        return np.concatenate(features)
