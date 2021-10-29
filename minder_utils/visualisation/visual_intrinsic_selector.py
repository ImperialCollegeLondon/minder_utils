import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import numpy as np
from minder_utils.configurations import config

sns.set()


def visualise(importance, datatype):
    plt.clf()
    plt.bar(np.arange(importance.shape[0]), importance)
    plt.xticks(np.arange(importance.shape[0]), config[datatype]['sensors'], rotation=90)
    plt.title(datatype)
    plt.tight_layout()
    plt.savefig('./results/visual/{}.png'.format(datatype))

