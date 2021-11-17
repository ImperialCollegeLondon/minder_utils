import pandas as pd


def compute_week_number(df):
    df = pd.to_datetime(df)
    return df.dt.isocalendar().week + (df.dt.isocalendar().year - 2000) * 100


