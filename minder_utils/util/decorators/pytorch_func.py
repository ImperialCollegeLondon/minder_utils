from functools import wraps
import torch


class pytorch_train:
    def __init__(self):
        pass

    def __call__(self, func):
        optimizer = torch.optim.Adam(self.model.parameters(), 3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            pass


