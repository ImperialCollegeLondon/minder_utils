from minder_utils.models.utils.early_stopping import EarlyStopping
from minder_utils.models import model_config
import torch
from minder_utils.models.feature_extractors.simclr.basic import ResNetSimCLR
from minder_utils.models.utils.util import get_device
from minder_utils.models.feature_extractors.partial_order.loss import Ranking
import numpy as np
import os


class Partial_Order:
    def __init__(self):
        self.early_stop = EarlyStopping()
        self.config = model_config.partial_order
        self.device = get_device()
        self.model = ResNetSimCLR(**self.config["model"])
        self.criterion = Ranking(**self.config["loss"])

    def _step(self, xi, xj, anchor):
        ris, zis = self.model(xi)
        rjs, zjs = self.model(xj)
        ras, zas = self.model(anchor)
        return self.criterion(zis, zjs, zas)

    def train(self, train_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), 3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)
        for epoch_counter in range(self.config['epochs']):
            for pre_anchor, anchor, post_anchor in train_loader:
                optimizer.zero_grad()
                loss = 0
                for idx_day in range(len(post_anchor) - 1):
                    loss += self._step(post_anchor[idx_day], post_anchor[idx_day + 1], anchor)
                    loss += self._step(pre_anchor[idx_day], pre_anchor[idx_day + 1], anchor)

                loss.backward()
                print('loss: ', loss.item(), end='\n')
                optimizer.step()
                scheduler.step()
                self.early_stop(loss.item(), self.model)
                if self.early_stop.early_stop:
                    break
            if self.early_stop.early_stop:
                break

    def test(self, test_loader, save_path=None):
        """
        :param test_loader: sample validated date only
        :return:
        """
        # validation steps
        with torch.no_grad():
            self.model.eval()
            features = []
            for pre_anchor, anchor, post_anchor in test_loader:
                feat, _ = self.model(anchor)
                features.append(feat.numpy())

        if save_path:
            np.save(os.path.join(save_path, 'partial_order.npy'), np.concatenate(features))

        return np.concatenate(features)

