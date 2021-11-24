from minder_utils.models.feature_extractors.simclr.basic import ResNetSimCLR
from minder_utils.models.feature_extractors.simclr.loss import NTXentLoss
import torch.nn.functional as F
from minder_utils.models.utils import Feature_extractor


class SimCLR(Feature_extractor):

    def __init__(self):
        super(SimCLR, self).__init__()
        self.nt_xent_criterion = NTXentLoss(self.device, self.config['loss']['temperature'],
                                            self.config['loss']['use_cosine_similarity'])
        self.model = ResNetSimCLR(**self.config["model"]).to(self.device)

    def step(self, data):
        (xis, xjs), _ = data
        ris, zis = self.model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = self.model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

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

    @staticmethod
    def which_data(data):
        return data[0]
