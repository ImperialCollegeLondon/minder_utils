import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from minder_utils.configurations import config

sns.set()


def visualise(importance, datatype):
    plt.clf()
    df = pd.DataFrame(importance)
    df = df.melt(var_name='Sensor', value_name='Importance')
    map_dict = dict(zip(np.arange(len(config[datatype]['sensors'])),
                        config[datatype]['sensors']))
    df.Sensor = df.Sensor.map(map_dict)
    sns.boxplot(x='Sensor', y='Importance', data=df)
    plt.title(datatype)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('./results/visual/{}.png'.format(datatype))
