import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from minder_utils.formatting.label import label_by_week
from minder_utils.feature_engineering import Feature_engineer
from minder_utils.feature_engineering.calculation import *
from minder_utils.util import formatting_plots
from minder_utils.formatting import Formatting

fe = Feature_engineer(Formatting())

sns.set()

att = 'bathroom_night'
figure_title = {
    'bathroom_night': 'Bathroom activity during the night',
    'bathroom_daytime': 'Bathroom activity during the day',
}

patient_id = 'JYN9EVX3wyv76VbubFPpUB'


def process_dataframe(df, week_shift=0):
    df = df[df.id == patient_id]
    map_dict = {i: j - week_shift for j, i in enumerate(df.week.sort_values().unique())}
    df.week = df.week.map(map_dict)
    return df


def visualise_flags(df):
    for v in [True, False]:
        data = df[df.valid == v]
        not_labelled = True
        for week in data.week.unique():
            if v is True:
                plt.axvline(week, 0, 0.17, color='red', label='UTI' if not_labelled else None)
                not_labelled = False
            elif v is False:
                plt.axvline(week, 0, 0.17, color='blue', label='not UTI' if not_labelled else None)
                not_labelled = False


@formatting_plots(figure_title[att])
def visualise_weekly_data(df):
    df = process_dataframe(df)
    sns.violinplot(data=df, x='week', y='value')
    visualise_flags(df)
    return df


@formatting_plots('P value, ' + figure_title[att])
def visualise_weekly_statistical_analysis(df, results):
    df = process_dataframe(df, 1)
    visualise_flags(df)
    data = results[patient_id]
    df = {'week': [], 'p_value': []}
    for idx, sta in enumerate(data):
        df['week'].append(idx + 1)
        df['p_value'].append(sta[1])
    sns.lineplot(df['week'], df['p_value'])


@formatting_plots('Body temperature')
def visualise_body_temperature(df):
    df = process_dataframe(df)
    visualise_flags(df)
    sns.lineplot(df.week, df.value)



def visualise_data_time_lineplot(time_array, values_array, name, fill_either_side_array=None, fig = None, ax = None):
    '''
    This function accepts a dataframe that has a ```'time'``` column and 
    and a ```'value'``` column.

    '''
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize = (10,6))

    ax.plot(time_array, values_array)
    
    if not fill_either_side_array is None:
        ax.fill_between(time_array, 
                        y1=values_array-fill_either_side_array, 
                        y2=values_array+fill_either_side_array,
                        alpha = 0.3)


    return fig, ax




def visualise_data_time_heatmap(data_plot, name, fig = None, ax = None):
    '''
    This function accepts a dataframe in which the columns are the days and 
    the rows are the aggregated times of the day.


    '''

    if ax is None:
        fig, axes = plt.subplots(1,1,figsize = (10,6))


    ax = sns.heatmap(data_plot.values, cmap = 'Blues', cbar_kws={'label': name})
    ax.invert_yaxis()

    x_tick_loc = np.arange(0, data_plot.shape[1], 90)
    ax.set_xticks(x_tick_loc + 0.5)
    ax.set_xticklabels(data_plot.columns.astype(str)[x_tick_loc].values)

    y_tick_loc = np.arange(0, data_plot.shape[0], 3)
    ax.set_yticks(y_tick_loc + 0.5)
    ax.set_yticklabels([pd.to_datetime(time).strftime("%H:%M") for time in data_plot.index.values[y_tick_loc]], rotation = 0)

    ax.set_xlabel('Day')
    ax.set_ylabel('Time of Day')

    return fig, ax











if __name__ == '__main__':
    results = weekly_compare(getattr(fe, att), kolmogorov_smirnov)
    df = label_by_week(getattr(fe, att))
    visualise_weekly_data(df)
    visualise_weekly_statistical_analysis(df)
    visualise_body_temperature(label_by_week(fe.body_temperature))
