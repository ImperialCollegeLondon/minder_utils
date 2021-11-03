import torch


class Ranking(torch.nn.Module):

    def __init__(self, delta, use_cosine_similarity):
        super(Ranking, self).__init__()
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.measure_similarity = self._get_similarity_function(use_cosine_similarity)
        self.delta = delta
        self.criterion = torch.nn.BCELoss(reduction='sum')
        if not use_cosine_similarity:
            dim = 64
            self.projector = torch.nn.Linear(dim, dim, bias=False)

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._metrics_similarity

    def _metrics_similarity(self, x, y):
        return torch.sum(torch.square(self.projector(x) - self.projector(y)), dim=1)

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, z_anchor):
        """
        :param zis: similar to anchor
        :param zjs: dissimilar to anchor
        :param z_anchor: anchor image
        :return:
        """
        s1 = self.measure_similarity(zis, z_anchor)
        s2 = self.measure_similarity(zjs, z_anchor)
        # loss = - torch.mean(torch.log(torch.mean(torch.clamp(s2 - s1 + self.delta, min=0, max=1.), dim=-1)))
        margin = torch.clamp(s2 - s1 + self.delta, min=0, max=1.)
        loss = self.criterion(margin, torch.zeros_like(margin))
        return loss
