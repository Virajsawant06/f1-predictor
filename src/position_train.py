import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from data_loading import get_raw_dataset
from dnf_features import create_features
from position_features import create_training_dataset
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

def train_test_split(X, y, df):
    train = df[df['year'] < 2022].copy()
    test = df[df['year'] >= 2022].copy()

    X_train = train[X.columns]
    y_train = train['positionOrder']

    X_test = test[X.columns]
    y_test = test['positionOrder']

    return X_train, X_test, y_train, y_test

df = get_raw_dataset("../data/raw")

df = create_training_dataset(df)

X, y = prepare_data_position(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, df)

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

xgb_reg = RandomizedSearchCV(
    XGBRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=20,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=1,
    random_state=42
)

xgb_reg.fit(X_train, y_train)
print("Best Hyperparameters:", xgb_reg.best_params_)
best_xgb = xgb_reg.best_estimator_
best_xgb_pred = best_xgb.predict(X_test)
print(f"Best XGBoost MAE: {mean_absolute_error(y_test, best_xgb_pred):.2f}")

# joblib.dump(
#     xgb_reg.best_estimator_,
#     "../models/position_model.pkl"
# )

def train_position_model(df):
    X, y = prepare_data_position(df)
    
    xgb_reg.fit(X, y)
    print("Best position params:", xgb_reg.best_params_)
    
    joblib.dump(xgb_reg.best_estimator_, "../models/retrained_pos.pkl")
    print("Position model saved!")
    
    return xgb_reg.best_estimator_