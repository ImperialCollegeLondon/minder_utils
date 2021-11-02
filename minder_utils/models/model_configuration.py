class Configuration:
    def __init__(self):
        pass

    @property
    def simclr(self):
        return {
            'model': {'base_model': 'resnet18',
                      'out_dim': 128},
            'epochs': 50,
            'eval_every_n_epochs': 1
        }

    @property
    def partial_order(self):
        return {
            'model': {'base_model': 'resnet18',
                      'out_dim': 128},
            'epochs': 50,
            'eval_every_n_epochs': 1
        }