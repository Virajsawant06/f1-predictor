import joblib
import pandas as pd
from data_loading import get_raw_dataset
from dnf_features import create_features
from position_features import create_training_dataset

FEATURES = [
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

model = joblib.load("../models/position_model.pkl")


def predict_position(features_dict):

    X = pd.DataFrame([features_dict])

    X = X[FEATURES]

    prediction = model.predict(X)[0]

    return {
        "predicted_position": round(float(prediction))
    }

sample = {
    'grid': 1,
    'qualifying_position': 1,
    'grid_penalty': 0,
    'driver_championship_position': 1,
    'driver_championship_points': 400,
    'driver_wins': 10,
    'constructor_championship_position': 1,
    'constructor_championship_points': 700,
    'constructor_wins': 15,
    'driver_experience': 200,
    'driver_avg_position': 2,
    'constructor_avg_position': 2,
    'year': 2025,
    'round': 10,
    'is_street_circuit': 0,
    'dnf_probability': 0.05
}

result = predict_position(sample)

print(result)