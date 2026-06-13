import joblib
import pandas as pd
from data_loading import get_raw_dataset
from dnf_features import create_features

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

model = joblib.load("../models/dnf_model.pkl")


def predict_dnf(features_dict):

    X = pd.DataFrame([features_dict])

    X = X[FEATURES]

    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)[0][1]

    return {
        "prediction": int(prediction),
        "dnf_probability": float(probability)
    }

sample = {
    'grid': 20,
    'qualifying_position': 20,

    'driver_championship_points': 0,
    'driver_championship_position': 20,
    'driver_wins': 0,

    'constructor_championship_points': 0,
    'constructor_championship_position': 10,
    'constructor_wins': 0,

    'year': 2025,
    'round': 24,

    'driver_experience': 0,

    'driver_dnf_rate': 1.0,
    'driver_dnf_rate_last10': 1.0,
    'driver_dnf_rate_last5': 1.0,

    'constructor_dnf_rate': 1.0,
    'constructor_dnf_rate_last10': 1.0,
    'constructor_dnf_rate_last5': 1.0,

    'circuit_dnf_rate': 1.0,
    'is_street_circuit': 1
}

result = predict_dnf(sample)



model2 = joblib.load("../models/dnf_model_on_training.pkl")

DNF_FEATURES = FEATURES

def add_dnf_probability(df):

    df = df.copy()

    df["dnf_probability"] = model2.predict_proba(
        df[DNF_FEATURES]
    )[:, 1]

    return df



