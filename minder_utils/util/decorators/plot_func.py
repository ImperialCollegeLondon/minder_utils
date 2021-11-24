from functools import wraps
from minder_utils.util import save_mkdir
import matplotlib.pyplot as plt
import os


def formatting_plots(title=None, save_path=None, rotation=90, legend=True):
    def plot_decorator(func):
        figure_title = func.__name__ if title is None else title

        @wraps(func)
        def wrapped_functions(*args, **kwargs):
            plt.clf()
            func(*args, **kwargs)
            plt.xticks(rotation=rotation)
            plt.suptitle(figure_title)
            if legend:
                plt.legend(loc='upper right')
            plt.tight_layout()
            if save_path is not None:
                save_mkdir(save_path)
                plt.savefig(os.path.join(save_path, figure_title + '.png'))
            plt.show()

        return wrapped_functions

    return plot_decorator
