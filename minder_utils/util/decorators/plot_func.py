from functools import wraps
import matplotlib.pyplot as plt
import os


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