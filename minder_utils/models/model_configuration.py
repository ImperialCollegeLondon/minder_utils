class Configuration:
    def __init__(self):
        self.input_dim = 336
        self.out_dim = 64

    @property
    def simclr(self):
        return {
            'model': {'base_model': 'basic',
                      'out_dim': self.out_dim,
                      'input_dim': self.input_dim},
            'epochs': 500,
            'eval_every_n_epochs': 1
        }

    @property
    def partial_order(self):
        return {
            'model': {'base_model': 'basic',
                      'out_dim': self.out_dim,
                      'input_dim': self.input_dim},
            'epochs': 500,
            'eval_every_n_epochs': 1,
            'loss': {'delta': 0.5,
                     'use_cosine_similarity': True}
        }

    @property
    def autoencoder(self):
        return {
            'model': {'base_model': 'conv',
                      'out_dim': self.out_dim,
                      'input_dim': self.input_dim},
            'epochs': 500,
        }
