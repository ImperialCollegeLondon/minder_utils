import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from minder_utils.util import formatting_plots
from minder_utils.formatting import Formatting
from minder_utils.feature_engineering import Feature_engineer
import pandas as pd
from minder_utils.configurations import visual_config, config
import matplotlib
import time
from .util import time_to_seconds
import datetime
import networkx as nx

from minder_utils.formatting import standardise_activity_data
from minder_utils.formatting import l2_norm

from minder_utils.formatting.label import label_by_week as week_label
from minder_utils.formatting.label import label_dataframe
from minder_utils.feature_engineering.calculation import build_p_matrix
from minder_utils.feature_engineering.adding_features import get_entropy_rate as calculate_entropy_rate
from minder_utils.feature_engineering.util import week_to_date, datetime_to_time





sns.set()


class Visualisation_Activity:
    def __init__(self, patient_id=None, date=None, valid=None):
        self.reset(patient_id, date, valid)

    @staticmethod
    def get_label_info():
        activity_data = label_dataframe(Formatting().activity_data)
        labelled_data = activity_data[activity_data.valid.isin([True, False])]
        labelled_data.time = labelled_data.time.dt.date
        labelled_data = labelled_data[['id', 'time', 'valid']].drop_duplicates()
        print(labelled_data)
        return labelled_data

    def reset(self, patient_id=None, date=None, valid=None):
        assert valid in [None, True, False], 'the valid must be in [None, True, False]'

        # select data according to patient id, date, valid
        activity_data = label_dataframe(Formatting().activity_data)

        if valid is not None:
            activity_data = activity_data[activity_data.valid == valid]

        if patient_id is None:
            activity_data = self.filter_dates(date, activity_data)
            activity_data = self.filter_patient(patient_id, activity_data)
        else:
            if patient_id != 'all':
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

    def top_5_hist_plot(self, fig=None, ax=None):

        data_plot = self.data['raw_data'][['time', 'location']].copy()

        unique_locations, counts = np.unique(data_plot['location'].values, return_counts=True)
        unique_locations_to_plot = unique_locations[np.argsort(counts)[::-1]][:5]

        data_plot['TOD'] = datetime_to_time(data_plot['time'])
        data_plot = data_plot[data_plot.location.isin(unique_locations_to_plot)]

        unique_locations = data_plot.location.unique()
        location_dict = {i: unique_locations[i] for i in range(len(unique_locations))}
        data_plot = data_plot.dropna().reset_index()

        sns.set_theme('talk')

        cp = [
            sns.color_palette("muted")[4],
            sns.color_palette("bright")[2],
            sns.color_palette('Paired', 9)[-2],  
            sns.color_palette('bright', desat=0.75)[-1],
            sns.color_palette("deep")[3]
        ]


        sns.set_palette(sns.color_palette(cp), len(unique_locations_to_plot))

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize = (12,8))

        ax = sns.histplot(data=data_plot.reset_index(), x = 'TOD', 
                        hue = 'location', ax=ax, legend=True, 
                        multiple='stack', bins = 100, hue_order = unique_locations_to_plot)


        xlim = (pd.to_datetime('1900-01-01 00:00:00'), pd.to_datetime('1900-01-02 00:00:00'))
        ax.set_xlim(xlim)

        hours = mdates.HourLocator(interval = 6)
        h_fmt = mdates.DateFormatter('%H:%M')

        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(h_fmt)

        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')

        ax.set_title('Stacked Frequency of Firings for the Top 5 Sensors')

        ax.tick_params(axis='x', rotation = 0)

        return fig, ax


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


if __name__ == '__main__':
    visual = Visualisation_Activity()




class Visualisation_Entropy:
    '''
    This class allows the user to visualise the entropy of the data.
    
    
    Arguments
    ---------

    - activity_data_list: list:
        A list of the ```pandas.DataFrames``` that will be used to calculate 
        the entropy and plots. Each dataframe should have an id, location and time
        column.
    
    - id: string or list, optional:
        A list or string that contains the ids that are to be plotted.
        Defaults to ```'all'```

    - data_list_names: list, optional:
        A list that has the names corresponding to the different datasets given as
        each element of ```activity_data_list```. If ```None```, names will be
        automatically generated.
        Defaults to ```None```.

    
    
    '''

    def __init__(self, activity_data_list, patient_id='all', data_list_names=None, week_or_day = 'week'):
        '''
        This function is used to initialise the class.
        '''
        
        if type(activity_data_list) == list:
            self.activity_data_list = activity_data_list
        else:
            self.activity_data_list = [activity_data_list]

        if not data_list_names is None:
            if type(data_list_names) == str:
                if len(self.activity_data_list) == 1:
                    self.data_list_names = data_list_names
                else:
                    raise TypeError('Please supply data_list_names as a list with as many' \
                                    'elements as activity_data_list')
            elif type(data_list_names) == list:
                if len(self.activity_data_list) == len(data_list_names):
                    self.data_list_names = data_list_names
                else:
                    raise TypeError('Please supply data_list_names as a list with as many' \
                                    'elements as activity_data_list')
            else:
                raise TypeError('Please supply data_list_names as a list with as many' \
                                'elements as activity_data_list')
        else:
            self.data_list_names = ['Set {}'.format(i + 1) for i in range(len(self.activity_data_list))]

        if type(patient_id) == str:
            if not patient_id == 'all':
                self.id = [patient_id]
            else:
                self.id = patient_id
        else:
            self.id = patient_id

        if not self.id == 'all':
            filtered_activity_data_list = []
            for data in self.activity_data_list:
                filtered_activity_data_list.append(data[data.id.isin(self.id)])

            self.activity_data_list = filtered_activity_data_list

        self.entropy_data_list = None
        self.entropy_data = None

        self.week_or_day = week_or_day

        return

    def _get_entropy_rate_data(self, by_flag=False):
        '''
        This is an internal function that is used to calculate the entropy and 
        filter the data based on the ```id``` given in the initialisation of the class.
        '''
        entropy_data_list = []
        for nd, data in enumerate(self.activity_data_list):

            movement_data = data[['id', 'time', 'location']]
            movement_data = movement_data.dropna(subset=['location'])

            movement_data = calculate_entropy_rate(movement_data, week_or_day = self.week_or_day)
            movement_data['data_label'] = self.data_list_names[nd]
            if self.week_or_day == 'week':
                movement_data = week_label(movement_data)
                movement_data['date'] = week_to_date(movement_data['week'])
            elif self.week_or_day == 'day':
                movement_data = movement_data.rename(columns = {'date':'time'})
                movement_data = label_dataframe(movement_data)
                movement_data = movement_data.rename(columns = {'time':'date'})

            entropy_data_list.append(movement_data)

        self.entropy_data_list = entropy_data_list
        self.entropy_data = pd.concat(entropy_data_list)

        return

    def _get_p_matrix_data(self, combine_data_list=True):
        '''
        This is an internal function that is used to calculate the p_matrix and 
        filter the data based on the ```id``` given in the initialisation of the class.
        '''
        if combine_data_list:
            data_full = pd.concat(self.activity_data_list)
            p_matrix, unique_locations = build_p_matrix(data_full['location'].values, return_events=True)
            return p_matrix, unique_locations

        else:
            p_matrix_list = []
            unique_locations_list = []
            for nd, data in enumerate(self.activity_data_list):
                p_matrix, unique_locations = build_p_matrix(data['location'].values, return_events=True)
                p_matrix_list.append(p_matrix)
                unique_locations_list.append(unique_locations)
            return p_matrix_list, unique_locations_list

    def plot_violin_entropy_rate(self, fig=None, ax=None, by_flag=False):
        '''
        This class allows the user to plot the entropy rate of each week in
        a violin plot.
        
        
        
        Arguments
        ---------
        
        - fig: matplotlib.pyplot.figure, optional:
            This is the figure to draw the plot on. If ```None```, then one will be created. 
            Defaults to ```None```.
        
        - ax: matplotlib.pyplot.axes, optional:
            This is the axes to draw the plot on. If ```None```, then one will be created. 
            Defaults to ```None```.

        - by_flag: bool, optional:
            Dictates whether the violin plots should be split on their flags.
            Defaults to ```False```.
        
        
        
        Returns
        --------
        
        - fig: matplotlib.pyplot.figure, optional:
            The figure that the axes are plotted on.
        
        - ax: matplotlib.pyplot.axes, optional:
            The axes that contain the graph.
        
        '''

        self._get_entropy_rate_data(by_flag=by_flag)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        if by_flag:
            ax = sns.violinplot(x='value', y='data_label',
                                data=self.entropy_data, ax=ax, inner='quartile',
                                split=True, hue='valid',
                                orient='h', cut=0, color='xkcd:light teal')

        else:
            ax = sns.violinplot(x='value', y='data_label',
                                data=self.entropy_data, ax=ax, inner='quartile',
                                orient='h', cut=0, color='xkcd:light teal')

        ax.set_ylabel('Dataset')
        ax.set_xlabel('Normalised Entropy Rate')
        ax.set_xlim(0, 1)
        ax.set_title('Entropy Rate Of PWLD Homes Each Week')

        return fig, ax

    def plot_boxplot_entropy_rate(self, fig=None, ax=None, by_flag=False):
        '''
        This class allows the user to plot the entropy rate of each week in
        a box plot.
        
        
        
        Arguments
        ---------
        
        - fig: matplotlib.pyplot.figure, optional:
            This is the figure to draw the plot on. If ```None```, then one will be created. 
            Defaults to ```None```.
        
        - ax: matplotlib.pyplot.axes, optional:
            This is the axes to draw the plot on. If ```None```, then one will be created. 
            Defaults to ```None```.

        - by_flag: bool, optional:
            Dictates whether the violin plots should be split on their flags.
            Defaults to ```False```.
        
        
        
        Returns
        --------
        
        - fig: matplotlib.pyplot.figure, optional:
            The figure that the axes are plotted on.
        
        - ax: matplotlib.pyplot.axes, optional:
            The axes that contain the graph.
        
        '''

        self._get_entropy_rate_data(by_flag=by_flag)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        if by_flag:
            ax = sns.boxplot(x='value', y='data_label',
                                data=self.entropy_data, hue ='valid', ax=ax,
                                orient='h', color='xkcd:light teal')

        else:
            ax = sns.boxplot(x='value', y='data_label',
                                data=self.entropy_data, ax=ax,
                                orient='h', color='xkcd:light teal')

        ax.set_ylabel('Dataset')
        ax.set_xlabel('Normalised Entropy Rate')
        ax.set_xlim(0, 1)
        ax.set_title('Entropy Rate Of PWLD Homes Each Week')

        return fig, ax


    def plot_line_entropy_rate(self, fig=None, ax=None):
        self._get_entropy_rate_data()

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 4))


        sns.lineplot(data=self.entropy_data, x='date', y='value', hue = 'id')
        ax.set_ylabel('Normalised Entropy Rate')
        ax.set_xlabel('Date')
        ax.set_ylim(0,1)

        ax.set_title('Weekly Entropy Rate')

        return fig, ax



    def plot_p_matrix(self, combine_data_list=True, fig=None, ax=None):
        '''
        This class allows the user to plot the stochastic matrix of the data 
        in a heatmap.
        
        
        
        Arguments
        ---------
        
        - combine_data_list: bool, optional:
            Dictates whether each element of ```activity_data_list``` will be plotted
            together or separately.
            Defaults to ```True```.

        - fig: matplotlib.pyplot.figure, optional:
            This is the figure to draw the plot on. If ```None```, then one will be created. 
            Defaults to ```None```.
        
        - ax: matplotlib.pyplot.axes, optional:
            This is the axes to draw the plot on. If ```None```, then one will be created. 
            Defaults to ```None```.
        
        
        
        Returns
        --------
        
        - fig: matplotlib.pyplot.figure, optional:
            The figure that the axes are plotted on.
        
        - ax: matplotlib.pyplot.axes, optional:
            The axes that contain the graph.
        
        '''

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        p_matrix, unique_locations = self._get_p_matrix_data(combine_data_list=True)

        if combine_data_list:

            ax = sns.heatmap(data=p_matrix, vmin=0, 
                            vmax=np.max(p_matrix), cmap='Blues', cbar=False, 
                            square=True)
            ax.invert_yaxis()

            ax.set_yticks(np.arange(len(unique_locations)) + 0.5)
            ax.set_yticklabels(unique_locations, rotation=0)

            ax.set_xticks(np.arange(len(unique_locations)) + 0.5)
            ax.set_xticklabels(unique_locations, rotation=90)

            ax.set_title('Stochastic Matrix Of Locations Visited')


        else:
            raise TypeError('combine_data_list=False is not currently supported.')

        return fig, ax

    def plot_p_matrix_graph(self, combine_data_list=True, fig=None, ax=None):
        '''
        Plots a directed graph using networkx that represents the stochastic matrix.


        Arguments
        ---------

        - combine_data_list: bool, optional:
            Dictates whether each element of ```activity_data_list``` will be plotted
            together or separately.
            Defaults to ```True```.

        - fig: matplotlib.pyplot.figure, optional:
            This is the figure to draw the plot on. If ```None```, then one will be created. 
            Defaults to ```None```.
        
        - ax: matplotlib.pyplot.axes, optional:
            This is the axes to draw the plot on. If ```None```, then one will be created. 
            Defaults to ```None```.

        Returns
        ---------

        
        - fig: matplotlib.pyplot.figure, optional:
            The figure that the axes are plotted on.
        
        - ax: matplotlib.pyplot.axes, optional:
            The axes that contain the graph.
            

        '''

        if combine_data_list:
            if ax is None:
                fig, ax = plt.subplots(1,1,figsize=(15,10))

            a_matrix, labels = self._get_p_matrix_data(combine_data_list=True)

            G = nx.convert_matrix.from_numpy_array(a_matrix, create_using=nx.DiGraph)
            G = nx.relabel.relabel_nodes(G, {i:labels[i] for i in range(len(labels))})

            pos=nx.shell_layout(G)
            nx.draw_networkx_nodes(G, pos=pos)


            edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())


            ax = nx.draw(G, pos, node_color='black', edgelist=edges, 
                            edge_color=weights, width=3.0, edge_cmap=plt.cm.Blues,  
                            arrowsize=30, ax = ax)

            for label in nx.nodes(G):
                plt.text(pos[label][0],pos[label][1]-0.1,s=label, 
                        bbox=dict(boxstyle="round", facecolor='blue', alpha=0.2),
                        horizontalalignment='center')


        else:
            raise TypeError('combine_data_list=False is not currently supported.')


        return fig, ax


class Visualisation_Bathroom():
    '''
    This class allows the user to produce visualisations 
    of the features engineered around bathroom visits.


    '''
    def __init__(self, patient_id='all'):

        self.fmg = Formatting()
        self.fe = Feature_engineer(self.fmg)

        if type(patient_id) == str:
            if patient_id == 'all':
                self.id = patient_id
            else:
                self.id = [patient_id]
        elif patient_id == list:
            self.id = patient_id
        else:
            raise TypeError("patient_id must be a string, 'all' or a list")

        return
    
    def _get_data(self, time_name, ma=False, delta=False):
        attr = 'bathroom'
        time_name_accept = ['daytime', 'night']
        if not time_name in time_name_accept:
            raise TypeError('Please use a time_name from {}'.format(time_name_accept))
        attr += '_' + time_name
        if ma:
            attr += '_' + 'ma'
            if delta:
                attr += '_' + 'delta'
        else:
            if delta:
                raise TypeError('Cannot use delta=True if ma=False')
        
        data = getattr(self.fe, attr)
        data['time'] = pd.to_datetime(data['time'])
        data = label_dataframe(data)

        self.attr = attr
        
        if not self.id == 'all':
            data = data[data.id.isin(self.id)]

        return data

    def plot_night(self, plot_type, ma=False, delta=False, by_flag=False, fig=None, ax=None):
        
        data = self._get_data('night', ma=ma, delta=delta)
        fig, ax = self._plot_data(data=data, plot_type=plot_type, by_flag=by_flag, fig=fig, ax=ax)
        ax.set_title(self.attr)
        
        return fig, ax


    def plot_day(self, plot_type, ma=False, delta=False, by_flag=False, fig=None, ax=None):
        
        data = self._get_data('daytime', ma=ma, delta=delta)
        fig, ax = self._plot_data(data=data, plot_type=plot_type, by_flag=by_flag, fig=fig, ax=ax)
        ax.set_title(self.attr)
        
        return fig, ax
    


    def _plot_data(self, data, plot_type, by_flag=False, fig=None, ax=None):
        
        data = data.dropna(subset=['value'])
        data['value'] = data['value'].astype(float)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        if plot_type=='boxplot':
            if by_flag:
                ax = sns.boxplot(x='value',
                                    data=data, y ='valid', ax=ax,
                                    orient='h', color='xkcd:light teal')
            else:
                ax = sns.boxplot(x='value',
                                    data=data, ax=ax,
                                    orient='h', color='xkcd:light teal')
            ax.set_xlabel('Value')
            
        elif plot_type == 'violin':
            if by_flag:
                ax = sns.violinplot(x='value', y = 'valid',
                                    data=data, ax=ax, inner='quartile',
                                    orient='h', cut=0, color='xkcd:light teal')

            else:
                ax = sns.violinplot(x='value',
                                    data=data, ax=ax, inner='quartile',
                                    orient='h', cut=0, color='xkcd:light teal')
            ax.set_xlabel('Value')

        elif plot_type == 'line':
            ax = sns.lineplot(x='time', y='value', data=data)
            ax.tick_params(axis='x', rotation=90)
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')


        return fig, ax
            


class Visualisation_Location():
    '''
    This class allows the user to produce visualisations 
    of the features engineered around bathroom visits.


    '''
    def __init__(self, location, patient_id='all'):

        self.fmg = Formatting()
        self.fe = Feature_engineer(self.fmg)
        self.location = location

        if type(patient_id) == str:
            if patient_id == 'all':
                self.id = patient_id
            else:
                self.id = [patient_id]
        elif patient_id == list:
            self.id = patient_id
        else:
            raise TypeError("patient_id must be a string, 'all' or a list")

        return
    
    def _get_data(self, ma=False, delta=False):
        attr = self.location
        attr += '_activity'
        if ma:
            attr += '_' + 'ma'
            if delta:
                attr += '_' + 'delta'
        else:
            if delta:
                raise TypeError('Cannot use delta=True if ma=False')
        
        data = getattr(self.fe, attr)
        data['time'] = pd.to_datetime(data['time'])
        data = label_dataframe(data)

        self.attr = attr
        
        if not self.id == 'all':
            data = data[data.id.isin(self.id)]

        return data

    def plot_data(self, plot_type, ma=False, delta=False, by_flag=False, fig=None, ax=None):
        
        data = self._get_data(ma=ma, delta=delta)
        fig, ax = self._plot_data(data=data, plot_type=plot_type, by_flag=by_flag, fig=fig, ax=ax)
        ax.set_title(self.attr)
        
        return fig, ax
    


    def _plot_data(self, data, plot_type, by_flag=False, fig=None, ax=None):
        
        data = data.dropna(subset=['value'])
        data['value'] = data['value'].astype(float)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        if plot_type=='boxplot':
            if by_flag:
                ax = sns.boxplot(x='value',
                                    data=data, y ='valid', ax=ax,
                                    orient='h', color='xkcd:light teal')
            else:
                ax = sns.boxplot(x='value',
                                    data=data, ax=ax,
                                    orient='h', color='xkcd:light teal')
            ax.set_xlabel('Value')
            
        elif plot_type == 'violin':
            if by_flag:
                ax = sns.violinplot(x='value', y = 'valid',
                                    data=data, ax=ax, inner='quartile',
                                    orient='h', cut=0, color='xkcd:light teal')

            else:
                ax = sns.violinplot(x='value',
                                    data=data, ax=ax, inner='quartile',
                                    orient='h', cut=0, color='xkcd:light teal')
            ax.set_xlabel('Value')

        elif plot_type == 'line':
            ax = sns.lineplot(x='time', y='value', data=data)
            ax.tick_params(axis='x', rotation=90)
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')


        return fig, ax
            
