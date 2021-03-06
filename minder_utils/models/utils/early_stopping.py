import numpy as np
import torch
import os
from minder_utils.util import save_mkdir


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, verbose=False, delta=0, path='./ckpt', save_model=False,
                 trace_func=print, **kwargs):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_model = save_model

    def __call__(self, val_loss, model, save_name):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save_model:
                self.save_checkpoint(val_loss, model, save_name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.trace_func('Training is stopped due to early stopping')
        else:
            self.best_score = score
            if self.save_model:
                self.save_checkpoint(val_loss, model, save_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_mkdir(self.path)
        torch.save(model.state_dict(), os.path.join(self.path, save_name))
        self.val_loss_min = val_loss
