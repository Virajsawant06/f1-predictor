import pandas as pd
import numpy as np 
from data_loading import get_raw_dataset



def create_target(df):

    df = df.copy()

    finished_statuses = df[df['status'].str.contains(r'^\+\d+ Lap', regex=True)]['statusId'].unique().tolist()
    finished_statuses.append(1)  

    df['DNF'] = (~df['statusId'].isin(finished_statuses)).astype(int)

    remove_statuses = ['Did not qualify', 'Did not prequalify', 'Withdrew', 'Excluded']
    df = df[~df['status'].isin(remove_statuses)]

    return df

def add_driver_experience(df):

    df = df.sort_values(['driverId', 'date'])

    df['driver_experience'] = df.groupby('driverId').cumcount()

    return df

def filter_data_from_2014(df):

    df = df[df['year'] >= 2014]

    return df

def add_constructor_dnf_rate(df):

    df = df.sort_values(['constructorId', 'date'])

    df['constructor_dnf_rate'] = df.groupby('constructorId')['DNF'].transform(
    lambda x: x.shift(1).rolling(20, min_periods=1).mean()
    )

    df['constructor_dnf_rate'] = df['constructor_dnf_rate'].fillna(0.28)

    return df

def add_constructor_dnf_rate_last10(df):

    df = df.sort_values(['constructorId', 'date'])

    df['constructor_dnf_rate_last10'] = df.groupby('constructorId')['DNF'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    df['constructor_dnf_rate_last10'] = df['constructor_dnf_rate_last10'].fillna(0.28)

    return df

def add_constructor_dnf_rate_last5(df):

    df = df.sort_values(['constructorId', 'date'])

    df['constructor_dnf_rate_last5'] = df.groupby('constructorId')['DNF'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    df['constructor_dnf_rate_last5'] = df['constructor_dnf_rate_last5'].fillna(0.28)

    return df

def add_driver_dnf_rate(df):

    df = df.sort_values(['driverId', 'date'])

    df['driver_dnf_rate'] = df.groupby('driverId')['DNF'].transform(
    lambda x: x.shift(1).rolling(20, min_periods=1).mean()
    )

    df['driver_dnf_rate'] = df['driver_dnf_rate'].fillna(0.28)

    return df

def add_driver_dnf_rate_last10(df):

    df = df.sort_values(['driverId', 'date'])

    df['driver_dnf_rate_last10'] = df.groupby('driverId')['DNF'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    df['driver_dnf_rate_last10'] = df['driver_dnf_rate_last10'].fillna(0.28)

    return df

def add_driver_dnf_rate_last5(df):

    df = df.sort_values(['driverId', 'date'])

    df['driver_dnf_rate_last5'] = df.groupby('driverId')['DNF'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    df['driver_dnf_rate_last5'] = df['driver_dnf_rate_last5'].fillna(0.28)

    return df

def add_circuit_dnf_rate(df):

    df = df.sort_values(['circuitId', 'date'])

    df['circuit_dnf_rate'] = df.groupby('circuitId')['DNF'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    df['circuit_dnf_rate'] = df['circuit_dnf_rate'].fillna(0.17)

    return df

def add_is_street_circuit(df):

    street_circuits = [
    'albert_park',
    'monaco',
    'valencia',
    'marina_bay',
    'baku',
    'jeddah',
    'miami',
    'vegas'
    ]
    df['is_street_circuit'] = df['circuitRef'].isin(street_circuits).astype(int)

    return df

def handle_missing_values(df):

    qualifying_position_fill = (
        df.groupby('raceId')['grid']
          .transform('max')
    )

    df['qualifying_position'] = (
        df['qualifying_position']
        .fillna(qualifying_position_fill)
    )

    df['driver_championship_points'] = (
        df['driver_championship_points']
        .fillna(0)
    )

    df['driver_championship_position'] = (
        df['driver_championship_position']
        .fillna(
            df['driver_championship_position'].median()
        )
    )

    df['driver_wins'] = (
        df['driver_wins']
        .fillna(0)
    )

    df['constructor_championship_points'] = (
        df['constructor_championship_points']
        .fillna(0)
    )

    df['constructor_championship_position'] = (
        df['constructor_championship_position']
        .fillna(
            df['constructor_championship_position'].median()
        )
    )

    df['constructor_wins'] = (
        df['constructor_wins']
        .fillna(0)
    )

    return df


def create_features(df):

    df = create_target(df)
    df = add_driver_experience(df)
    df = filter_data_from_2014(df)
    df = add_constructor_dnf_rate(df)
    df = add_constructor_dnf_rate_last10(df)
    df = add_constructor_dnf_rate_last5(df)
    df = add_driver_dnf_rate(df)
    df = add_driver_dnf_rate_last10(df)
    df = add_driver_dnf_rate_last5(df)
    df = add_circuit_dnf_rate(df)
    df = add_is_street_circuit(df)
    df = handle_missing_values(df)
    return df

if __name__ == '__main__':
    df = get_raw_dataset("../data/raw")

    df = create_features(df)