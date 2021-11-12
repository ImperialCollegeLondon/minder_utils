import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, base_model, input_dim, out_dim):
        super(Encoder, self).__init__()
        # Encoder
        if base_model == 'conv':
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 8, (2, 2)),
                nn.Tanh(),
                nn.Conv2d(8, 16, (2, 2)),
                nn.Tanh(),
                nn.Conv2d(16, 8, (2, 2)),
                nn.Tanh(),
                nn.Conv2d(8, 3, (2, 2)),
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
        codes = self.encoder(inputs)
        return codes


class Decoder(nn.Module):
    def __init__(self, base_model, input_dim, out_dim):
        super(Decoder, self).__init__()
        # Decoder
        if base_model == 'conv':
            self.decoder = nn.Sequential(
                nn.Conv2d(3, 8, (2, 2)),
                nn.Tanh(),
                nn.Conv2d(8, 16, (2, 2)),
                nn.Tanh(),
                nn.Conv2d(16, 8, (2, 2)),
                nn.Tanh(),
                nn.Conv2d(8, 3, (2, 2)),
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
    def __init__(self, base_model, input_dim, out_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(base_model, input_dim, out_dim)
        self.decoder = Decoder(base_model, input_dim, out_dim)

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)
        return codes, decoded
