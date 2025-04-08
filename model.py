import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model


#load the model and scaler
model = load_model('onion_yield_model_v3.keras')
scaler = joblib.load('scaler_v2.pkl')
df = pd.read_csv('combined_climate_yield_data_V1.csv')

FEATURE_ORDER = ['tempmax', 'tempmin', 'temp', 'humidity', 'rainfall', 'sunshine_hours']

def preprocess_input(data):
    scaled_values = scaler.transform([[data['rainfall'], data['sunshine_hours']]])
    final_input = [
        data['tempmax'], data['tempmin'], data['temp'], data['humidity'],
        scaled_values[0][0], scaled_values[0][1]
    ]
    return np.array(final_input).reshape(1, -1)

def predict_yield(data):
    preprocessed = preprocess_input(data)
    prediction = model.predict(preprocessed)
    return round(float(prediction[0][0]), 4)

def get_random_sample_and_predict():
    sample = df.sample(n=1).iloc[0]
    year = sample['Year']
    input_data = sample[FEATURE_ORDER].to_dict()
    predicted = predict_yield(input_data)
    return {
        "input": input_data,
        "predicted_yield": float(predicted),
        "actual_yield": round(sample['avg_yield'], 4),
        "year": int(year)
    }

