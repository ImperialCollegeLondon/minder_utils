from minder_utils.formatting import Formatting
from minder_utils.feature_engineering.calculation import build_p_matrix
import matplotlib.pyplot as plt
from hmmviz import TransGraph
import matplotlib.patches as mpatches
import os
import numpy as np
import pandas as pd
import calendar


locations = ['']
formater = Formatting()
df = formater.activity_data[formater.activity_data.id == formater.activity_data.id.unique()[0]]


df = df[df.location.isin(['bedroom1', 'bathroom1', 'hallway', 'kitchen',
                          'front door', 'back door'])]
df.time = pd.to_datetime(df.time)

for month in df.time.dt.month.unique():

    sequence = df[df.time.dt.month == month].location.values
    T = pd.crosstab(
        pd.Series(sequence[:-1], name='From'),
        pd.Series(sequence[1:], name='To'),
    )
    graph = TransGraph(T)

    # looks best on square figures/axes
    fig = plt.figure(figsize=(18, 18))

    nodelabels = {i: i.split('1')[0] for i in np.unique(sequence)}
    colors = {f'{j}': f'C{i}' for i,j in zip(range(len(np.unique(sequence))), np.unique(sequence))}

    graph.draw(edgecolors=colors, nodecolors=colors, edgelabels=True,
               edgewidths=2, nodefontsize=25, edgefontsize=13, nodelabels=False
    )
    patches = []
    for key in colors:
        patches.append(mpatches.Patch(color=colors[key], label=key))
    plt.legend(handles=patches, prop={'size': 20})
    plt.savefig('{}.png'.format(calendar.month_name[month]))
