import pandas as pd


def load_clean_data(data_path):

    results = pd.read_csv(f'{data_path}/results.csv')
    status = pd.read_csv(f'{data_path}/status.csv')
    races = pd.read_csv(f'{data_path}/races.csv')
    drivers = pd.read_csv(f'{data_path}/drivers.csv')
    constructors = pd.read_csv(f'{data_path}/constructors.csv')
    qualifying = pd.read_csv(f'{data_path}/qualifying.csv')
    circuits = pd.read_csv(f'{data_path}/circuits.csv')
    driver_standings = pd.read_csv(f'{data_path}/driver_standings.csv')
    constructor_standings = pd.read_csv(f'{data_path}/constructor_standings.csv')

    results_clean = results[['resultId', 'raceId', 'driverId', 'constructorId', 
                            'grid', 'position', 'positionOrder', 'points', 
                            'laps', 'statusId', 'fastestLap', 'rank', 
                            'fastestLapTime', 'fastestLapSpeed']]

    races_clean = races[['raceId', 'circuitId', 'name', 'year', 'round', 'date']]
    races_clean = races_clean.rename(columns={'name': 'race_name'})

    circuits_clean = circuits[['circuitId', 'circuitRef', 'country', 'lat', 'lng']]

    drivers_clean = drivers[['driverId', 'driverRef', 'forename', 'surname', 
                            'nationality', 'dob']]
    drivers_clean = drivers_clean.rename(columns={'forename': 'driver_forename', 
                                                    'surname': 'driver_surname', 
                                                    'nationality': 'driver_nationality'})

    constructors_clean = constructors[['constructorId', 'constructorRef', 
                                    'name', 'nationality']]
    constructors_clean = constructors_clean.rename(columns={'name': 'constructor_name', 
                                                            'nationality': 'constructor_nationality'})

    qualifying_clean = qualifying[['raceId', 'driverId', 'number', 'position']]
    qualifying_clean = qualifying_clean.rename(columns={'position': 'qualifying_position',
                                                        'number': 'driver_number'})

    status_clean = status[['statusId', 'status']]

    driver_standings_clean = driver_standings[['raceId', 'driverId', 'points', 'position', 'wins']]
    driver_standings_clean = driver_standings_clean.rename(columns={'points': 'driver_championship_points',
                                                                    'position': 'driver_championship_position',
                                                                    'wins': 'driver_wins'})

    constructor_standings_clean = constructor_standings[['raceId', 'constructorId', 'points', 'position', 'wins']]
    constructor_standings_clean = constructor_standings_clean.rename(columns={'points': 'constructor_championship_points',
                                                                            'position': 'constructor_championship_position',
                                                                            'wins': 'constructor_wins'})

    return (results_clean, status_clean, races_clean, drivers_clean, constructors_clean, qualifying_clean, circuits_clean, driver_standings_clean, constructor_standings_clean)

def merge_data(results_clean, status_clean, races_clean, drivers_clean, constructors_clean, qualifying_clean, circuits_clean, driver_standings_clean, constructor_standings_clean):
    df = pd.merge(results_clean, races_clean, on='raceId', how='left')
    df = pd.merge(df, circuits_clean, on='circuitId', how='left')
    df = pd.merge(df, drivers_clean, on='driverId', how='left')
    df = pd.merge(df, constructors_clean, on='constructorId', how='left')
    df = pd.merge(df, qualifying_clean, on=['raceId', 'driverId'], how='left')
    df = pd.merge(df, status_clean, on='statusId', how='left')
    df = pd.merge(df, driver_standings_clean, on=['raceId', 'driverId'], how='left', suffixes=('', '_driver_standing'))
    df = pd.merge(df, constructor_standings_clean, on=['raceId', 'constructorId'], how='left', suffixes=('', '_constructor_standing'))

    return df

def get_raw_dataset(data_path):

    dataframes = load_clean_data(data_path)

    df = merge_data(*dataframes)

    return df
