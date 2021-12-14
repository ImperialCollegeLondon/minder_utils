import torch
import numpy as np
from sklearn.base import clone as sklearn_reset
import inspect
from sklearn.preprocessing import StandardScaler


def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on:", device)
    return device


def train_test_scale(X_train, X_test):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test



class SklearnModelWrapper:
    '''
    This function allows you to wrap an sklearn model with the 
    method ```.reset()``` which resets the model and its
    learned parameters, but keeps the set initialised parameters.
    It also attempts to resolve the issue in which the y input
    to the model is 2d, but the model itself only accepts 1d
    arrays in y.

    Arguments
    ---------

    - model: sklearn class:
        This is an initialised sklearn class.

    '''
    def __init__(self, model, model_type = 'nn'):
        self.model = model
        self.model_type = model_type

    def __getattr__(self, name):
        attr = getattr(self.model, name)
        
        if callable(attr):
            def wrapper(*args, **kwargs):
                try:
                    out = attr(*args, **kwargs)
                except ValueError as e:
                    if 'y should be a 1d array' in str(e):
                        new_args = []
                        y_pos = list(inspect.signature(attr).parameters.keys()).index('y')
                        if len(list(args)) > y_pos:
                            new_y = np.argmax(args[y_pos], axis = 1)
                            new_args = [(args[na] if na != y_pos else new_y) for na in range(len(args))]
                        else:
                            kwargs['y'] = np.argmax(kwargs['y'], axis = 1)
                            new_args = args

                        out = attr(*new_args, **kwargs)
                    else:
                        raise e
                return out
            return wrapper

        else:
            return attr


    def reset(self, *args, **kwargs):
        self.model = sklearn_reset(self.model)