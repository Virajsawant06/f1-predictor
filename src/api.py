from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from dnf_predict import predict_dnf
from position_predict import predict_position

app = FastAPI(title="F1 Race Predictior")

dnf_model = joblib.load("../models/dnf_model.pkl")
position_model = joblib.load("../models/position_model.pkl")

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

@app.get("/")
def home():
    return {"message": "F1 API is Running"}

class DNFInput(BaseModel):
    grid: int
    qualifying_position: float
    driver_championship_points: float
    driver_championship_position: float
    driver_wins: float
    constructor_championship_points: float
    constructor_championship_position: float
    constructor_wins: float
    year: int
    round: int
    driver_experience: int
    constructor_dnf_rate: float
    constructor_dnf_rate_last10: float
    constructor_dnf_rate_last5: float
    driver_dnf_rate: float
    driver_dnf_rate_last10: float
    driver_dnf_rate_last5: float
    circuit_dnf_rate: float
    is_street_circuit: int

class PositionInput(BaseModel):
    driver_name: str = "unknown"
    grid: int
    qualifying_position: float
    grid_penalty: float
    driver_championship_points: float
    driver_championship_position: float
    driver_wins: float
    constructor_championship_points: float
    constructor_championship_position: float
    constructor_wins: float
    year: int
    round: int
    driver_experience: int
    driver_avg_position: float
    constructor_avg_position: float
    is_street_circuit: int
    dnf_probability: float

class RaceInput(BaseModel):
    driver_name: str = "unknown"
    grid: int
    qualifying_position: float
    grid_penalty: float
    driver_championship_points: float
    driver_championship_position: float
    driver_wins: float
    constructor_championship_points: float
    constructor_championship_position: float
    constructor_wins: float
    year: int
    round: int
    driver_experience: int
    driver_avg_position: float
    constructor_avg_position: float
    is_street_circuit: int
    dnf_probability: float = 0.0  # ← optional, default 0
    # DNF model features
    constructor_dnf_rate: float
    constructor_dnf_rate_last10: float
    constructor_dnf_rate_last5: float
    driver_dnf_rate: float
    driver_dnf_rate_last10: float
    driver_dnf_rate_last5: float
    circuit_dnf_rate: float

@app.post("/predict/dnf")
def predict_dnf_endpoint(data: DNFInput):
    result = predict_dnf(data.model_dump())
    return result

@app.post("/predict/position")
def predict_position_endpoint(data: PositionInput):
    result = predict_position(data.model_dump())
    return result

@app.post("/predict/race")
def predict_race(race_data: list[RaceInput]):
    if len(race_data) < 2:
        return {"error": "At least 2 drivers required"}
    
    predictions = []
    
    for driver in race_data:

        data = driver.model_dump()
        driver_name = data.pop('driver_name')
        
        dnf_keys = [
        'grid', 'qualifying_position', 'driver_championship_points',
        'driver_championship_position', 'driver_wins',
        'constructor_championship_points', 'constructor_championship_position',
        'constructor_wins', 'year', 'round', 'driver_experience',
        'constructor_dnf_rate', 'constructor_dnf_rate_last10',
        'constructor_dnf_rate_last5', 'driver_dnf_rate',
        'driver_dnf_rate_last10', 'driver_dnf_rate_last5',
        'circuit_dnf_rate', 'is_street_circuit'
        ]

        dnf_features = {}
        for key in dnf_keys:
            dnf_features[key] = data[key]
        
        dnf_result = predict_dnf(dnf_features)
        
        
        data['dnf_probability'] = dnf_result['dnf_probability']
        
        # position_keys = [ 
        # 'grid',
        # 'qualifying_position',
        # 'grid_penalty',
        # 'driver_championship_position',
        # 'driver_championship_points',
        # 'driver_wins',
        # 'constructor_championship_position',
        # 'constructor_championship_points',
        # 'constructor_wins',
        # 'driver_experience',
        # 'driver_avg_position',
        # 'constructor_avg_position',
        # 'year',
        # 'round',
        # 'is_street_circuit',
        # 'dnf_probability']

        # position_features = {}
        # for key in position_keys:
        #     position_features[key] = data[key]
        
        result = predict_position(data)
        predictions.append({
            "driver": driver_name,
            "score": result['predicted_position'],
            "dnf_probability": round(dnf_result['dnf_probability'] * 100, 1)
        })
    
    
    scores = [p['score'] for p in predictions]
    ranks = pd.Series(scores).rank(method='first').astype(int).tolist()
    
    results = []
    for i, pred in enumerate(predictions):
        results.append({
            "driver": pred['driver'],
            "predicted_position": ranks[i],
            "dnf_probability": pred['dnf_probability']
        })
    
    return sorted(results, key=lambda x: x['predicted_position'])