from minder_utils.models.utils.early_stopping import EarlyStopping
from minder_utils.models import model_config
import torch
from minder_utils.models.feature_extractors.simclr.basic import ResNetSimCLR


class Partial_Order:
    def __init__(self):
        self.early_stop = EarlyStopping()
        self.config = model_config.partial_order

    def _setp(self):
        pass

    def train(self, train_loader):
        model = ResNetSimCLR(**self.config["model"])
        optimizer = torch.optim.Adam(model.parameters(), 3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)
        for epoch_counter in range(self.config['epochs']):
            for data in train_loader:
                optimizer.zero_grad()
