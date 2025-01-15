import pandas as pd

def preprocess_dataset1(df):
    # Preprocessing steps for Dataset 1
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['hour'] = df['datetime'].dt.hour
    return df

def preprocess_dataset2(df):
    # Preprocessing steps for Dataset 2
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month_name()
    df['hour'] = df['datetime'].dt.hour
    return df