import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from minder_utils.util import formatting_plots
from minder_utils.formatting import Formatting
import pandas as pd
from minder_utils.configurations import visual_config, config
import matplotlib
import time
from .util import time_to_seconds
import datetime
from minder_utils.formatting import standardise_activity_data
from minder_utils.formatting import l2_norm

sns.set()


class Visualisation_Activity:
    def __init__(self, patient_id=None, date=None, valid=None):
        self.reset(patient_id, date, valid)

    @staticmethod
    def get_label_info():
        activity_data = Formatting().activity_data
        labelled_data = activity_data[activity_data.valid.isin([True, False])]
        labelled_data.time = labelled_data.time.dt.date
        labelled_data = labelled_data[['id', 'time', 'valid']].drop_duplicates()
        print(labelled_data)
        return labelled_data

    def reset(self, patient_id=None, date=None, valid=None):
        assert valid in [None, True, False], 'the valid must be in [None, True, False]'

        # select data according to patient id, date, valid
        activity_data = Formatting().activity_data

        if valid is not None:
            activity_data = activity_data[activity_data.valid == valid]

        if patient_id is None:
            activity_data = self.filter_dates(date, activity_data)
            activity_data = self.filter_patient(patient_id, activity_data)
        else:
            activity_data = self.filter_patient(patient_id, activity_data)
            activity_data = self.filter_dates(date, activity_data)

        self.data = {'raw_data': activity_data}

        # Raw data
        if isinstance(date, datetime.date):
            self.data['raw_data'] = activity_data[activity_data.time.dt.date == date]
        elif isinstance(date, list):
            self.data['raw_data'] = activity_data[activity_data.time.dt.date.isin(date)]

        # add columns for visualisation
        self.data['raw_data']['seconds'] = self.data['raw_data'].time.dt.time.astype(str).apply(time_to_seconds)

        mapping = {}
        for idx, sensor in enumerate(config['activity']['sensors']):
            mapping[sensor] = idx + 1
        self.data['raw_data']['activity'] = self.data['raw_data'].location.map(mapping) \
                                            * self.data['raw_data'].value

        # Aggregate data
        self.data['agg'] = standardise_activity_data(self.data['raw_data'][activity_data.columns])

        # Normalise data
        self.data['normalise'] = self.data['agg'].copy()
        self.data['normalise'][config['activity']['sensors']] = self.data['normalise'][
            config['activity']['sensors']].apply(l2_norm)

    @formatting_plots(save_path=visual_config['activity']['save_path'], rotation=0, legend=False)
    def raw_data(self):
        self.subplots(self.data['raw_data'], self._visual_scatter)

    @formatting_plots(save_path=visual_config['activity']['save_path'], rotation=0, legend=False)
    def aggregated_data(self):
        self.subplots(self.data['agg'], self._visual_heatmap)

    @formatting_plots(save_path=visual_config['activity']['save_path'], rotation=0, legend=False)
    def normalised_data(self):
        self.subplots(self.data['normalise'], self._visual_heatmap)

    @staticmethod
    def filter_patient(patient_id, activity_data):
        if patient_id is None:
            patient_id = np.random.choice(activity_data.id.unique())
        if isinstance(patient_id, str):
            activity_data = activity_data[activity_data.id == patient_id]
        elif isinstance(patient_id, list):
            activity_data = activity_data[activity_data.id.isin(patient_id)]
        return activity_data

    @staticmethod
    def filter_dates(date, activity_data):
        available_dates = activity_data.time.dt.date.unique()
        if date is None:
            date = np.random.choice(available_dates)
        else:
            if isinstance(date, list):
                visual_dates = []
                for day in date:
                    v_date = pd.to_datetime(day, format='%Y-%m-%d').date()
                    visual_dates.append(v_date)
                    assert v_date in available_dates, 'The date provided is not in the dataset'
                date = visual_dates
            elif isinstance(date, str):
                date = pd.to_datetime(date, format='%Y-%m-%d').date()
                assert date in available_dates, 'The date provided is not in the dataset'
            else:
                raise TypeError('The date only accepts str or list as input')
        # Raw data
        if isinstance(date, datetime.date):
            activity_data = activity_data[activity_data.time.dt.date == date]
        elif isinstance(date, list):
            activity_data = activity_data[activity_data.time.dt.date.isin(date)]
        return activity_data

    @staticmethod
    def _visual_scatter(data, ax):
        yticks = config['activity']['sensors']
        ax.set_yticks(range(1, len(yticks) + 1))
        ax.set_yticklabels(yticks)
        sns.scatterplot(x='seconds', y='activity', marker="$\circ$", ec="face", data=data, ax=ax)
        formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%H', time.gmtime(ms)))
        ax.xaxis.set_major_formatter(formatter)

    @staticmethod
    def _visual_heatmap(data, ax):
        yticks = config['activity']['sensors']
        ax.set_yticks(range(1, len(yticks) + 1))
        ax.set_yticklabels(yticks)
        sns.heatmap(data[yticks].transpose().astype(float), ax=ax)
        formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%H', time.gmtime(ms * 3600)))
        ax.xaxis.set_major_formatter(formatter)

    @staticmethod
    def subplots(df, func):
        rows = df.id.unique()
        cols = df.time.dt.date.unique()
        width, height = plt.gcf().get_size_inches()
        width += len(cols)
        height += len(rows)
        fig, axes = plt.subplots(len(rows), len(cols), sharex=True, sharey=True, figsize=(width, height))
        mappings = {}
        for x_axis in range(len(rows)):
            for y_axis in range(len(cols)):
                data = df[(df.id == rows[x_axis]) & (df.time.dt.date == cols[y_axis])]
                try:
                    func(data, axes[x_axis, y_axis])
                    ax = axes[x_axis, y_axis]
                except TypeError:
                    func(data, axes)
                    ax = axes
                except IndexError:
                    indices = np.max([x_axis, y_axis])
                    func(data, axes[indices])
                    ax = axes[indices]

                ax.set_xlabel(None)
                ax.set_ylabel('Participant {}'.format(x_axis))
                ax.set_title(cols[y_axis])
                mappings[x_axis] = rows[x_axis]

        fig.supxlabel('Time')
        fig.supylabel('Location')
        print(mappings)


if __name__ == '__main__':
    visual = Visualisation_Activity()
