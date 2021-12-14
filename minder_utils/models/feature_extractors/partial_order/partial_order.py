from .basic import Partial_Order_Models
from .loss import Ranking
from minder_utils.models.utils import Feature_extractor
from minder_utils.dataloader import Partial_Order_Loader


class Partial_Order(Feature_extractor):
    def __init__(self):
        super(Partial_Order, self).__init__()
        self.model = Partial_Order_Models(**self.config["model"])
        self.criterion = Ranking(**self.config["loss"])

    def _custom_loader(self, data):
        X, y = data
        return Partial_Order_Loader(X, y, **self.config['loader'])

    def step(self, data):
        pre_anchor, anchor, post_anchor = data
        loss = 0
        for idx_day in range(len(post_anchor) - 1):
            loss += self._step(post_anchor[idx_day], post_anchor[idx_day + 1], anchor)
            loss += self._step(pre_anchor[idx_day], pre_anchor[idx_day + 1], anchor)
        return loss

    def _step(self, xi, xj, anchor):
        ris, zis = self.model(xi)
        rjs, zjs = self.model(xj)
        ras, zas = self.model(anchor)
        return self.criterion(zis, zjs, zas)

    @staticmethod
    def which_data(data):
        return data[0]
