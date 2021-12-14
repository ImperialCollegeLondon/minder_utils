import torch.nn as nn
import torch.nn.functional as F


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
                nn.Tanh(),
                nn.Flatten(),
                nn.Linear(input_dim, latent_dim)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Flatten(),
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


class Partial_Order_Models(nn.Module):
    def __init__(self, base_model, input_dim, out_dim, latent_dim, **kwargs):
        super(Partial_Order_Models, self).__init__()
        self.features = Encoder(base_model, input_dim, latent_dim)

        # projection MLP
        self.l1 = nn.Linear(latent_dim, latent_dim)
        self.l2 = nn.Linear(latent_dim, out_dim)

    def forward(self, x):
        h = self.features(x)
        h = nn.Flatten()(h)

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x
