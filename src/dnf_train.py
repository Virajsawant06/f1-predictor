import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from data_loading import get_raw_dataset
from dnf_features import create_features
from sklearn.metrics import f1_score, accuracy_score, classification_report
import joblib


def prepare_data(df):
    features = [
        'grid',
        'qualifying_position', 
        'driver_championship_points',
        'driver_championship_position',
        'driver_wins',
        'constructor_championship_points',
        'constructor_championship_position',
        'constructor_wins',
        'year',
        'round',
        'driver_experience',
        'constructor_dnf_rate',
        'constructor_dnf_rate_last10',
        'constructor_dnf_rate_last5',
        'driver_dnf_rate',
        'driver_dnf_rate_last10',
        'driver_dnf_rate_last5',
        'circuit_dnf_rate',
        'is_street_circuit'
    ]

    X = df[features]
    y = df['DNF']

    return X, y


def train_test_split(X, y, df):
    train = df[df['year'] < 2022].copy()
    test = df[df['year'] >= 2022].copy()

    X_train = train[X.columns]
    y_train = train['DNF']

    X_test = test[X.columns]
    y_test = test['DNF']

    return X_train, X_test, y_train, y_test


df = get_raw_dataset("../data/raw")

df = create_features(df)

X, y = prepare_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, df)


param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [4, 6, 8, 10, 15]
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=0),
    param_grid,
    n_iter=50,
    cv=5,
    scoring='f1',
    n_jobs=1,
    random_state=0
)

xgb_search.fit(X_train, y_train)
print("Best XGBoost Params:", xgb_search.best_params_)
best_xgb = xgb_search.best_estimator_
xgb_pred = best_xgb.predict(X_test)
print("Best XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
print("Best XGBoost F1:", f1_score(y_test, xgb_pred))
print("Classification Report:")
print(classification_report(y_test, xgb_pred))

# joblib.dump(
#     xgb_search.best_estimator_,
#     "../models/dnf_model.pkl"
# )

def train_dnf_model(df):
    X, y = prepare_data(df)
    
    xgb_search.fit(X, y)
    print("Best DNF params:", xgb_search.best_params_)
    
    joblib.dump(xgb_search.best_estimator_, "../models/retrained_dnf.pkl")
    print("DNF model saved!")
    
    return xgb_search.best_estimator_