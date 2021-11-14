from functools import wraps
import matplotlib.pyplot as plt
import os
from minder_utils.util.util import save_mkdir, save_file, load_file


class load_save:
    def __init__(self, save_path, save_name=None, verbose=True, refresh=False):
        self.save_path = save_path
        self.file_name = save_name
        self.verbose = verbose
        self.refresh = refresh

    def __call__(self, func):
        self.file_name = func.__name__ if self.file_name is None else self.file_name

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            if self.refresh:
                self.print_func(func, 'start to refresh the data')
                data = func(*args, **kwargs)
                save_file(data, self.save_path, self.file_name)
            else:
                try:
                    data = load_file(self.save_path, self.file_name)
                    self.print_func(func, 'loading processed data')
                except FileNotFoundError:
                    save_mkdir(self.save_path)
                    self.print_func(func, 'processing the data')
                    data = func(*args, **kwargs)
                    save_file(data, self.save_path, self.file_name)
            return data

        return wrapped_function

    def print_func(self, func, message):
        if self.verbose:
            print(str(func.__name__).ljust(20, ' '), message)


def formatting_plots(title, save_path=None):
    def plot_decorator(func):
        @wraps(func)
        def wrapped_functions(*args, **kwargs):
            plt.clf()
            func(*args, **kwargs)
            plt.xticks(rotation=90)
            plt.title(title)
            plt.legend(loc='upper right')
            plt.tight_layout()
            if save_path is not None:
                plt.savefig(os.path.join(save_path, title + '.png'))
            else:
                plt.show()
        return wrapped_functions
    return plot_decorator


__all__ = ['load_save']
