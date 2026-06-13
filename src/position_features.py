import pandas as pd
import numpy as np
from data_loading import get_raw_dataset
from dnf_features import create_features
from dnf_predict import add_dnf_probability
import joblib

def add_driver_avg_position(df):

    df = df.copy()

    df = df.sort_values(['driverId', 'date'])

    df['driver_avg_position'] = df.groupby('driverId')['positionOrder'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df['driver_avg_position'] = df['driver_avg_position'].fillna(10)

    return df

def add_constructor_avg_position(df):
    df = df.sort_values(['constructorId', 'date'])

    df['constructor_avg_position'] = df.groupby('constructorId')['positionOrder'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df['constructor_avg_position'] = df['constructor_avg_position'].fillna(10)

    return df

def add_grid_penalty(df):

    df['grid_penalty'] = df['qualifying_position'] - df['grid']
    
    return df

def add_position_features(df):

    df = add_driver_avg_position(df)
    df = add_constructor_avg_position(df)
    df = add_grid_penalty(df)

    return df

def create_training_dataset(df):
    df = create_features(df)
    df = add_dnf_probability(df)
    df = add_position_features(df)

    return df

if __name__ == '__main__':
    df = get_raw_dataset("../data/raw")

    df = create_training_dataset(df)




