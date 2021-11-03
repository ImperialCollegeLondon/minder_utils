class Configuration:
    def __init__(self):
        pass

    @property
    def simclr(self):
        return {
            'model': {'base_model': 'basic',
                      'out_dim': 64},
            'epochs': 500,
            'eval_every_n_epochs': 1
        }

    @property
    def partial_order(self):
        return {
            'model': {'base_model': 'basic',
                      'out_dim': 64},
            'epochs': 500,
            'eval_every_n_epochs': 1,
            'loss': {'delta': 0.5,
                     'use_cosine_similarity': True}
        }