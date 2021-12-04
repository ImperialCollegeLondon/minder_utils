import matplotlib.pyplot as plt
import seaborn as sns
from minder_utils.util import formatting_plots
from minder_utils.configurations import visual_config, config

sns.set_theme(style="ticks", palette="pastel")


class Visual_Evaluation:
    def __init__(self, dataframe=None, importance=None):
        self.results = dataframe
        self.importance = importance

    def reset(self, dataframe=None, importance=None):
        self.results = dataframe
        self.importance = importance

    @formatting_plots('Evaluation Results', save_path=visual_config['evaluation']['save_path'], rotation=0, legend=True)
    def boxplot(self):
        sns.set(font_scale=3.0)
        fig, axes = plt.subplots(2, 2, figsize=(30, 30), sharey=True)
        for idx, metric in enumerate(['sensitivity', 'specificity', 'acc', 'f1']):
            g_ = sns.boxplot(ax=axes[idx // 2, idx % 2], data=self.results, x='model', y=metric, hue='feature_type')
            if idx < 3:
                g_.legend_.remove()

    @formatting_plots('Importance', save_path=visual_config['evaluation']['save_path'], rotation=90, legend=False)
    def importance_bar(self):
        sns.set(font_scale=1.5)
        self.importance['importance'] *= 100 / self.importance['importance'].max()
        sns.barplot(x='importance', y='sensors', data=self.importance)
