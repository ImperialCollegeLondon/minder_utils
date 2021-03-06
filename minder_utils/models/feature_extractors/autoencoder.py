import torch.nn as nn
from minder_utils.models.utils import Feature_extractor


class AutoEncoder(Feature_extractor):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(**self.config['model'])
        self.decoder = Decoder(**self.config['model'])
        self.model = nn.Sequential(self.encoder, self.decoder)
        self.criterion = nn.BCELoss() if self.config['loss']['func'] == 'bce' else nn.MSELoss()

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)
        return codes, decoded

    def step(self, data):
        data, label = data
        if self.config['model']['base_model'] in ['nn']:
            data = data.view(data.size(0), -1)
        return self.criterion(self.decoder(self.encoder(data)), data)


class Encoder(nn.Module):
    def __init__(self, base_model, input_dim, latent_dim):
        super(Encoder, self).__init__()
        # Encoder
        if base_model == 'conv':
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=2, padding=1),
                nn.Tanh(),
                nn.Conv2d(8, 16, kernel_size=2, padding=1),
                nn.Tanh(),
                nn.Conv2d(16, 8, kernel_size=2),
                nn.Tanh(),
                nn.Conv2d(8, 3, kernel_size=2),
                nn.Tanh()
            )
        else:
            self.encoder = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(input_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, latent_dim),
                nn.Tanh()
            )

    def forward(self, inputs):
        codes = self.encoder(inputs)
        return codes


class Decoder(nn.Module):
    def __init__(self, base_model, input_dim, latent_dim):
        super(Decoder, self).__init__()
        # Decoder
        if base_model == 'conv':
            self.decoder = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=2, padding=1),
                nn.Tanh(),
                nn.Conv2d(8, 16, kernel_size=2, padding=1),
                nn.Tanh(),
                nn.Conv2d(16, 8, kernel_size=2),
                nn.Tanh(),
                nn.Conv2d(8, 3, kernel_size=2),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 128),
                nn.Tanh(),
                nn.Linear(128, 256),
                nn.Tanh(),
                nn.Linear(256, input_dim),
                nn.Sigmoid()
            )

    def forward(self, inputs):
        outputs = self.decoder(inputs)
        return outputs