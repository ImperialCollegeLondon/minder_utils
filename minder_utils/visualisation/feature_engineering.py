import matplotlib.pyplot as plt
import seaborn as sns
from minder_utils.formatting.label import label_by_week
from minder_utils.feature_engineering import Feature_engineer
from minder_utils.feature_engineering.calculation import *
from minder_utils.util.decorators import formatting_plots
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


if __name__ == '__main__':
    results = weekly_compare(getattr(fe, att), kolmogorov_smirnov)
    df = label_by_week(getattr(fe, att))
    visualise_weekly_data(df)
    visualise_weekly_statistical_analysis(df)
    visualise_body_temperature(label_by_week(fe.body_temperature))
