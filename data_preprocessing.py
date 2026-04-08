import pandas as pd

def load_data(path):
    cols = ['unit', 'time', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    data = pd.read_csv(path, sep='\s+', header=None)
    data = data.iloc[:, :26]
    data.columns = cols
    
    return data


def add_rul(df):
    max_cycle = df.groupby('unit')['time'].max().reset_index()
    max_cycle.columns = ['unit', 'max_time']
    
    df = df.merge(max_cycle, on='unit')
    df['RUL'] = df['max_time'] - df['time']
    
    return df


def drop_useless_columns(df):
    cols_to_drop = [
        'op3', 'sensor_1', 'sensor_5',
        'sensor_10', 'sensor_16',
        'sensor_18', 'sensor_19'
    ]
    
    return df.drop(columns=cols_to_drop)