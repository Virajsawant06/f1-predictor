import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import joblib


def prepare_data_position(df):
    position_features = [
        'grid',
        'qualifying_position',
        'grid_penalty',
        'driver_championship_position',
        'driver_championship_points',
        'driver_wins',
        'constructor_championship_position',
        'constructor_championship_points',
        'constructor_wins',
        'driver_experience',
        'driver_avg_position',
        'constructor_avg_position',
        'year',
        'round',
        'is_street_circuit',
        'dnf_probability'
    ]

    X = df[position_features]
    y = df['positionOrder']

    return X, y


param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}


def train_position_model(df):
    X, y = prepare_data_position(df)

    xgb_reg = RandomizedSearchCV(
        XGBRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=20,
        scoring='neg_mean_absolute_error',
        cv=3,
        verbose=1,
        random_state=42
    )

    xgb_reg.fit(X, y)
    print("Best position params:", xgb_reg.best_params_)

    joblib.dump(xgb_reg.best_estimator_, "../models/position_model.pkl")
    print("Position model saved!")

    return xgb_reg.best_estimator_