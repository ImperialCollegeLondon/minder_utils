import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from minder_utils.util import formatting_plots
from minder_utils.configurations import visual_config

sns.set()


class Visual_Sleep:
    def __init__(self, path, style='age', filename='imperial_dementia_20211026'):
        '''
        Visualise the sleep data
        Parameters
        ----------
        path: str, path to the sleep data
        style: str, plot style
            - age: lineplot, hue = age
            - joint:  lineplot, hue = age, style = Sleep Time
            - face: facegrid
            - re: relation plot

        '''
        self.config = visual_config['sleep']
        self.style = style
        if 'imperial' in filename:
            self.data = pd.read_csv(os.path.join(path, filename + '.csv'), delimiter=';')
        else:
            self.data = pd.read_csv(os.path.join(path, filename + '.csv'))

        # Divide the data by time
        self.data.start_date = pd.to_datetime(self.data.start_date)
        self.data['Sleep Time'] = 'Late'
        index = pd.DatetimeIndex(self.data.start_date)
        self.data['Sleep Time'].iloc[index.indexer_between_time('10:00', '21:00')] = 'Early'
        if 'imperial' in filename:
            self.data['age'] = 2021 - pd.to_datetime(self.data['birthdate']).dt.year
            self.data = self.data[self.data['age'] >= 60]
        # Group by ages
        self.data.age[self.data.age <= 50] = 0
        self.data.age[(self.data.age > 50) & (self.data.age <= 60)] = 1
        self.data.age[(self.data.age > 60) & (self.data.age <= 70)] = 2
        self.data.age[(self.data.age > 70) & (self.data.age <= 80)] = 3
        self.data.age[self.data.age > 80] = 4
        mapping = {
            0: '<=50', 1: '50-60', 2: '60-70', 3: '70-80', 4: '>80'
        }
        self.data.age = self.data.age.map(mapping)

        new_cols = []
        for col in self.data.columns:
            append = True
            for ele in self.config['stages']:
                if col in ele:
                    new_cols.append(ele)
                    append = False
            if append:
                new_cols.append(col)
        self.data.columns = new_cols

        df = self.data[self.config['stages']]
        for col in self.config['stages']:
            if col not in ['Sleep Time', 'age', 'user_id']:
                df = self.filter_df(df, col)
                df[col] /= 3600
        df['Sleep'] = df['light_duration (s)'] + df['deep_duration (s)'] + df['rem_duration (s)']
        df = df[['user_id', 'awake_duration (s)', 'Sleep Time', 'age', 'Sleep']]
        df = df.melt(id_vars=['user_id', 'Sleep Time', 'age'], var_name='State', value_name='Duration (H)')
        mapping = {
            'awake_duration (s)': 'Awake in bed',
            'Sleep': 'Sleep'
        }
        df['State'] = df['State'].map(mapping)
        self.df = df

    @formatting_plots(save_path=visual_config['sleep']['save_path'], rotation=90, legend=False)
    def lineplot(self):
        self.plt_func(sns.lineplot)

    @formatting_plots(save_path=visual_config['sleep']['save_path'], rotation=90, legend=False)
    def violinplot(self):
        self.plt_func(sns.violinplot)

    @formatting_plots(title='Duration of different states', save_path=visual_config['sleep']['save_path'], rotation=0, legend=False)
    def boxplot_separate(self):
        self.plt_func(sns.boxplot)

    @formatting_plots(title='Duration of different states', save_path=visual_config['sleep']['save_path'], rotation=0, legend=False)
    def boxplot_joint(self):
        style = self.style
        self.style = 'no_hue'
        self.plt_func(sns.boxplot)
        self.style = style

    @staticmethod
    def filter_df(df, col, width=1.5):
        # Computing IQR
        new_df = []
        for age in df.age.unique():
            Q1 = df[df.age == age][col].quantile(0.25)
            Q3 = df[df.age == age][col].quantile(0.75)
            IQR = Q3 - Q1
            indices = (df[df.age == age][col] >= Q1 - width * IQR) & (df[df.age == age][col] <= Q3 + width * IQR)
            new_df.append(df[df.age == age].loc[indices])
        # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
        return pd.concat(new_df)

    def plt_func(self, func, x_name='State', y_name='Duration (H)', hue_name='age'):
        if self.style == 'age':
            length = len(self.df[hue_name].unique())
            func(x=x_name, y=y_name, hue=hue_name, data=self.df, hue_order=self.config['hue_order'][-length:])
        elif self.style == 'joint':
            try:
                func(x=x_name, y=y_name, hue=hue_name, style='Sleep Time', data=self.df,
                     hue_order=self.config['hue_order'])
            except TypeError:
                func(x=x_name, y=y_name, hue=hue_name, data=self.df,
                     hue_order=self.config['hue_order'])
        elif self.style == 'face':
            g = sns.FacetGrid(self.df, col=hue_name, row='Sleep Time', col_order=self.config['hue_order'])
            g.map(func, x_name, y_name)
            for axes in g.axes.flat:
                _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
        elif self.style == 're':
            sns.relplot(
                data=self.df,
                x=x_name, y=y_name,
                hue="Sleep Time", col=hue_name,
                kind="line",
            )
        else:
            func(x=x_name, y=y_name, data=self.df, hue_order=self.config['hue_order'])

    @formatting_plots(save_path=visual_config['sleep']['save_path'], rotation=90, legend=False)
    def visual_phy(self):
        df = self.data[self.config['phy_stages']]
        for col in self.config['phy_stages']:
            if col not in ['age']:
                df[col] = df[col].apply(lambda x: float('.'.join(str(x).split(','))))
                df[col] = self.filter_df(df, col)
                df[col] /= df[col].max()
        df = df.melt(id_vars='age', var_name='Other Data', value_name='Value')
        sns.boxplot(x='Other Data', y='Value', hue='age', hue_order=self.config['hue_order'], data=df)

    @formatting_plots(save_path=visual_config['sleep']['save_path'], rotation=90, legend=False,
                      title='Time of participants went to bed')
    def visualise_counts(self):
        df = self.data[['age', 'Sleep Time', 'start_date']]
        df.start_date = pd.to_datetime(df.start_date)
        df['Time'] = df.start_date.dt.hour
        df['Percentage'] = 1
        df = df.groupby(by=['Sleep Time', 'age', 'Time'])['Percentage'].sum().reset_index()
        for a in df.age.unique():
            df['Percentage'][df.age == a] /= sum(df[df.age == a]['Percentage'])

        sns.lineplot(x='Time', y='Percentage', hue='age', linestyle='--', hue_order=self.config['hue_order'], data=df)
        plt.ylim(0, 0.35)

        ticks = []
        for i in range(24):
            ticks.append('{}.00'.format(str(i).zfill(2)))
        plt.xticks(range(24), ticks)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               fancybox=True, shadow=True, ncol=5)


if __name__ == '__main__':
    vs = Visual_Sleep('/Users/mozzie/Desktop/DATA.nosync/sleep_mat', 'age')
    vs.visualise_counts()
    vs.boxplot_joint()
    # vs.boxplot_separate()

    vs_with = Visual_Sleep('/Users/mozzie/Desktop/DATA.nosync/sleep_mat', 'age', filename='withings_sleep_dataset')

    # Joint, Controlled age group
    minder = vs.df
    minder['Dataset'] = 'Dementia'

    withings = vs_with.df

    withings_ages = dict(withings.age.value_counts())
    minder_ages = dict(minder.age.value_counts())
    min_times = 100000
    for age in minder_ages:
        times = withings_ages[age] / minder_ages[age]
        if times < min_times:
            min_times = times
    withings_df = []
    mappings = {}
    for age in minder_ages:
        num = minder_ages[age] * min_times
        mappings[age] = age + ' (' + str(round(minder_ages[age] / len(minder) * 100))[:2] + '%' + ')'
        withings_df.append(withings[withings.age == age].sample(n=int(num)))
    withings = pd.concat(withings_df)
    withings['Dataset'] = 'Aged Matched Control Group'
    withings.age = withings.age.map(mappings)
    minder.age = minder.age.map(mappings)

    df = pd.concat([minder, withings])
    sns.boxplot(x='State', y='Duration (H)', hue='Dataset', data=df)
    plt.savefig('joint.png')

    plt.clf()
    g = sns.FacetGrid(df, col='age', col_order=['60-70 (59%)', '70-80 (27%)', '>80 (14%)'])
    g.map(sns.boxplot, 'State', 'Duration (H)', 'Dataset')
    g.set(xlabel=None)
    plt.legend(loc='center left', bbox_to_anchor=(-1.4, -0.2),
               fancybox=True, shadow=True, ncol=5)
    plt.savefig('joint_face.png')