from minder_utils.models.feature_extractors.simclr.basic import ResNetSimCLR
from minder_utils.models.feature_extractors.partial_order.loss import Ranking
from minder_utils.models.utils import Feature_extractor


class Partial_Order(Feature_extractor):
    def __init__(self):
        super(Partial_Order, self).__init__()
        self.model = ResNetSimCLR(**self.config["model"])
        self.criterion = Ranking(**self.config["loss"])

    def step(self, data):
        pre_anchor, anchor, post_anchor = data
        loss = 0
        for idx_day in range(len(post_anchor) - 1):
            loss += self._step(post_anchor[idx_day], post_anchor[idx_day + 1], anchor)
            loss += self._step(pre_anchor[idx_day], pre_anchor[idx_day + 1], anchor)
        return loss

    @staticmethod
    def which_data(data):
        pre_anchor, anchor, post_anchor = data
        return anchor

    def _step(self, xi, xj, anchor):
        ris, zis = self.model(xi)
        rjs, zjs = self.model(xj)
        ras, zas = self.model(anchor)
        return self.criterion(zis, zjs, zas)
