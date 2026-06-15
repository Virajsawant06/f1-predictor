import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import joblib


def prepare_data(df):
    features = [
        'grid', 'qualifying_position', 'driver_championship_points',
        'driver_championship_position', 'driver_wins',
        'constructor_championship_points', 'constructor_championship_position',
        'constructor_wins', 'year', 'round', 'driver_experience',
        'constructor_dnf_rate', 'constructor_dnf_rate_last10',
        'constructor_dnf_rate_last5', 'driver_dnf_rate',
        'driver_dnf_rate_last10', 'driver_dnf_rate_last5',
        'circuit_dnf_rate', 'is_street_circuit'
    ]
    X = df[features]
    y = df['DNF']
    return X, y


param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [4, 6, 8, 10, 15]
}


def train_dnf_model(df):
    X, y = prepare_data(df)

    xgb_search = RandomizedSearchCV(
        XGBClassifier(random_state=0),
        param_grid,
        n_iter=50,
        cv=5,
        scoring='f1',
        n_jobs=1,
        random_state=0
    )

    xgb_search.fit(X, y)
    print("Best DNF params:", xgb_search.best_params_)

    best_xgb = xgb_search.best_estimator_

    calibrated_model = CalibratedClassifierCV(best_xgb, cv=5, method='isotonic')
    calibrated_model.fit(X, y)

    joblib.dump(calibrated_model, "../models/dnf_model.pkl")
    print("Calibrated DNF model saved!")

    return calibrated_model 