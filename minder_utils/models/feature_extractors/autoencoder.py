import torch.nn as nn
import torch
from minder_utils.models.utils.early_stopping import EarlyStopping
from minder_utils.models import model_config
import numpy as np
import os


class Encoder(nn.Module):
    def __init__(self, base_model, input_dim, out_dim):
        super(Encoder, self).__init__()
        # Encoder
        if base_model == 'conv':
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 8, 2, padding=1),
                nn.Tanh(),
                nn.Conv2d(8, 16, 2, padding=1),
                nn.Tanh(),
                nn.Conv2d(16, 8, 2),
                nn.Tanh(),
                nn.Conv2d(8, 3, 2),
                nn.Tanh()
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, out_dim),
                nn.Tanh()
            )

    def forward(self, inputs):
        codes = self.encoder(inputs[0])
        return codes


class Decoder(nn.Module):
    def __init__(self, base_model, input_dim, out_dim):
        super(Decoder, self).__init__()
        # Decoder
        if base_model == 'conv':
            self.decoder = nn.Sequential(
                nn.Conv2d(3, 8, 2, padding=1),
                nn.Tanh(),
                nn.Conv2d(8, 16, 2, padding=1),
                nn.Tanh(),
                nn.Conv2d(16, 8, 2),
                nn.Tanh(),
                nn.Conv2d(8, 3, 2),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(2, 64),
                nn.Tanh(),
                nn.Linear(64, 128),
                nn.Tanh(),
                nn.Linear(128, 256),
                nn.Tanh(),
                nn.Linear(256, 784),
                nn.Sigmoid()
            )

    def forward(self, inputs):
        outputs = self.decoder(inputs)
        return outputs


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.config = model_config.autoencoder
        self.encoder = Encoder(self.config['model']['base_model'], self.config['model']['input_dim'], self.config['model']['out_dim'])
        self.decoder = Decoder(self.config['model']['base_model'], self.config['model']['input_dim'], self.config['model']['out_dim'])
        self.early_stop = EarlyStopping()
        self.criterion = nn.BCELoss() if self.config['loss'] == 'bce' else nn.MSELoss()

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)
        return codes, decoded

    def fit(self, train_loader):
        parameters = list(self.encoder.parameters()) + (list(self.decoder.parameters()))
        optimizer = torch.optim.Adam(parameters, 3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)
        for epoch_counter in range(self.config['epochs']):
            for data in train_loader:
                optimizer.zero_grad()
                loss = self.criterion(self.decoder(self.encoder(data)), data[0])
                loss.backward()
                print('loss: ', loss.item(), end='\n')
                optimizer.step()
                scheduler.step()
                self.early_stop(loss.item(), self.encoder)
                if self.early_stop.early_stop:
                    break
            if self.early_stop.early_stop:
                break

    def predict(self, test_loader, save_path=None):
        """
        :param test_loader: sample validated date only
        :return:
        """
        # validation steps
        with torch.no_grad():
            self.encoder.eval()
            features = []
            for data in test_loader:
                feat = self.encoder(data)
                features.append(feat.numpy())

        if save_path:
            np.save(os.path.join(save_path, 'autoencoder.npy'), np.concatenate(features))

        return np.concatenate(features)
