from scipy.stats import ks_2samp


def weekly_compare(df, func, num_previous_week=1):
    '''
    Function to compare the values of each patient in current week to previous weeks
    Args:
        df: Dataframe, it should contains at least three columns, which are ['id', 'week', 'value'],
            where the id is the patient ids, week is the numeric numbers got from dt.week, value is the
            sensor readings.
        func: function, used to compare the difference between current week and previous week.
        num_previous_week: int, optional, default is 1, number of previous weeks

    Returns:
        results: dictionary, key is the patient id, value is a list containing the values calculated by func.

    '''
    assert num_previous_week >= 1, 'num_previous_week must be equal or greater than 1'
    num_weeks = df.week.sort_values().unique()
    results = {}
    for p_id in df.id.unique():
        results[p_id] = []
    for idx, week in enumerate(num_weeks):
        if idx < num_previous_week:
            continue
        current_week = df[df.week == week]
        previous_week = df[df.week.isin([week - i for i in range(1, num_previous_week + 1)])]
        for p_id in df.id.unique():
            previous_patient_data = previous_week[previous_week.id == p_id].value.to_numpy()
            current_patient_data = current_week[current_week.id == p_id].value.to_numpy()
            if current_patient_data.shape[0] == 0 or previous_patient_data.shape[0] == 0:
                continue
            try:
                results[p_id].append(func(current_patient_data, previous_patient_data))
            except ValueError:
                pass
    return results


def threshold_compare(df, func='>', threshold=36):
    '''
    Function to filter the dataframe by threshold
    Args:
        df:
        func:
        threshold:

    Returns:

    '''
    if func == '>':
        return df[df.value > threshold]
    elif func == '<':
        return df[df.value < threshold]


def kolmogorov_smirnov(freq1, freq2):
    return ks_2samp(freq1, freq2)


