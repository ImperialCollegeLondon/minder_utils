import torch
from minder_utils.models.feature_extractors.simclr.basic import ResNetSimCLR
from minder_utils.models.feature_extractors.simclr.loss import NTXentLoss
from minder_utils.models import model_config
from minder_utils.models.utils.early_stopping import EarlyStopping
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, batch_size):
        self.device = self._get_device()
        self.nt_xent_criterion = NTXentLoss(self.device, batch_size=batch_size, temperature=0.5, use_cosine_similarity=True)
        self.early_stop = EarlyStopping()
        self.model = None
        self.config = model_config.simclr

    @staticmethod
    def _get_device():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, n_iter):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self, train_loader):
        model = ResNetSimCLR(**self.config["model"]).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), 3e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            if self.early_stop.early_stop:
                break
            for (xis, xjs), _ in train_loader:
                optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, n_iter)

                loss.backward()

                optimizer.step()
                n_iter += 1

                self.early_stop(loss.item(), model)
                print(loss.item())
                if self.early_stop.early_stop:
                    break

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
        self.model = model

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs), _ in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
