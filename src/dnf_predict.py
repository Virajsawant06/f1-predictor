import joblib
import pandas as pd

FEATURES = [
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

DNF_FEATURES = FEATURES

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load("../models/dnf_model.pkl")
    return _model


def predict_dnf(features_dict):
    model = get_model()
    X = pd.DataFrame([features_dict])
    X = X[FEATURES]

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "prediction": int(prediction),
        "dnf_probability": float(probability)
    }


def add_dnf_probability(df):
    model = get_model()
    df = df.copy()
    df["dnf_probability"] = model.predict_proba(df[DNF_FEATURES])[:, 1]
    return df