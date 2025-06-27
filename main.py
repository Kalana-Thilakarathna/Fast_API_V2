from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import warnings
from fastapi.middleware.cors import CORSMiddleware


warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="Onion Yield Prediction API",
    description="API for predicting onion yields using Prophet models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_DIR = "models"
DATA_DIR = "data"
CLIMATE_FEATURES = ['tempmax', 'tempmin', 'temp', 'humidity', 'rainfall', 'sunshine_hours']
GROWING_SEASON_MONTHS = [5, 6, 7, 8]  # May to August

# Load models on startup
models = {}

def load_models():
    """Load all Prophet models"""
    try:
        # Load yield prediction model
        yield_model_path = os.path.join(MODEL_DIR, "prophet_yield_model.pkl")
        with open(yield_model_path, 'rb') as f:
            models['yield'] = pickle.load(f)
        
        # Load climate prediction models (updated to use new models)
        for feature in CLIMATE_FEATURES:
            model_path = os.path.join(MODEL_DIR, f"{feature}_prophet_model.pkl")
            with open(model_path, 'rb') as f:
                models[feature] = pickle.load(f)
        
        print("All models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def add_growing_season_indicator(df):
    """Add growing season indicator to DataFrame"""
    df = df.copy()
    df['growing_season'] = df['ds'].dt.month.isin(GROWING_SEASON_MONTHS)
    return df

def validate_and_constrain_predictions(predictions, variable_name, historical_data):
    """Apply realistic constraints to predictions based on historical data"""
    if historical_data is None or len(historical_data) == 0:
        return predictions
    
    hist_stats = historical_data[variable_name].describe()
    
    # Define realistic bounds based on historical data and climate knowledge
    constraints = {
        'tempmax': (hist_stats['min'] - 2, hist_stats['max'] + 2),
        'tempmin': (hist_stats['min'] - 2, hist_stats['max'] + 2),
        'temp': (hist_stats['min'] - 2, hist_stats['max'] + 2),
        'humidity': (max(0, hist_stats['min'] - 0.5), hist_stats['max'] + 0.5),
        'rainfall': (0, hist_stats['max'] * 2),  # Rainfall can't be negative
        'sunshine_hours': (max(0, hist_stats['min'] - 1), min(24, hist_stats['max'] + 1))
    }
    
    if variable_name in constraints:
        lower_bound, upper_bound = constraints[variable_name]
        predictions = np.clip(predictions, lower_bound, upper_bound)
    
    return predictions

def load_data():
    """Load historical climate and yield data"""
    try:
        historical_data_path = os.path.join(DATA_DIR, "yala_season_climate_yield_data.csv")
        historical_data = pd.read_csv(historical_data_path)
        historical_data['datetime'] = pd.to_datetime(historical_data['datetime'])
        
        forecasted_data_path = os.path.join(DATA_DIR, "forecasted_climate_data_growing_season.csv")
        forecasted_data = None
        if os.path.exists(forecasted_data_path):
            forecasted_data = pd.read_csv(forecasted_data_path)
            forecasted_data['datetime'] = pd.to_datetime(forecasted_data['datetime'])
        
        return historical_data, forecasted_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# Load models and data on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models and data on startup"""
    success = load_models()
    if not success:
        print("Warning: Some models failed to load")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Onion Yield Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "yield_predictions": "/predict-yields",
            "update_climate_forecast": "/update-climate-forecast"
        }
    }

@app.get("/predict-yields")
async def predict_yields():
    """
    First endpoint: Get actual and predicted yields from 2004-2023 plus 2 years of predictions
    """
    try:
        # Load historical data
        historical_data, forecasted_climate = load_data()
        if historical_data is None:
            raise HTTPException(status_code=500, detail="Failed to load historical data")
        
        # Prepare historical data for prediction
        historical_yields = historical_data.groupby('Year').agg({
            'Average Yield (Metric Tons/Hectare)': 'first',
            'datetime': 'first'
        }).reset_index()
        historical_yields = historical_yields.rename(columns={
            'Average Yield (Metric Tons/Hectare)': 'actual_yield'
        })
        
        # Prepare data for Prophet prediction (historical + forecast)
        prophet_data = []
        
        # Add historical data
        for _, row in historical_data.iterrows():
            prophet_data.append({
                'ds': row['datetime'],
                'tempmax': row['tempmax'],
                'tempmin': row['tempmin'],
                'temp': row['temp'],
                'humidity': row['humidity'],
                'rainfall': row['rainfall'],
                'sunshine_hours': row['sunshine_hours']
            })
        
        # Add forecasted climate data
        if forecasted_climate is not None:
            for _, row in forecasted_climate.iterrows():
                prophet_data.append({
                    'ds': row['datetime'],
                    'tempmax': row['tempmax'],
                    'tempmin': row['tempmin'],
                    'temp': row['temp'],
                    'humidity': row['humidity'],
                    'rainfall': row['rainfall'],
                    'sunshine_hours': row['sunshine_hours']
                })
        
        prophet_df = pd.DataFrame(prophet_data)
        
        # Add growing season indicator if needed for yield model
        prophet_df = add_growing_season_indicator(prophet_df)
        
        # Make predictions using the yield model
        if 'yield' not in models:
            raise HTTPException(status_code=500, detail="Yield prediction model not loaded")
        
        # Check if yield model requires growing_season column
        try:
            forecast = models['yield'].predict(prophet_df)
        except Exception as e:
            if 'growing_season' in str(e):
                # If yield model doesn't need growing_season, remove it
                prophet_df_simple = prophet_df.drop(columns=['growing_season'], errors='ignore')
                forecast = models['yield'].predict(prophet_df_simple)
            else:
                raise e
        
        # Combine historical and predicted data
        results = []
        
        # Group by year for historical data (2004-2023)
        historical_years = historical_data['Year'].unique()
        for year in sorted(historical_years):
            year_data = historical_data[historical_data['Year'] == year]
            actual_yield = year_data['Average Yield (Metric Tons/Hectare)'].iloc[0]
            
            # Get predicted yield for this year
            year_forecast = forecast[prophet_df['ds'].dt.year == year]
            if not year_forecast.empty:
                predicted_yield = year_forecast['yhat'].mean()
            else:
                predicted_yield = None
            
            results.append({
                'year': int(year),
                'actual_yield': float(actual_yield),
                'predicted_yield': float(predicted_yield) if predicted_yield is not None else None,
                'type': 'historical'
            })
        
        # Add future predictions (2024-2025)
        future_years = [2024, 2025]
        for year in future_years:
            year_forecast = forecast[prophet_df['ds'].dt.year == year]
            if not year_forecast.empty:
                predicted_yield = year_forecast['yhat'].mean()
                results.append({
                    'year': year,
                    'actual_yield': None,
                    'predicted_yield': float(predicted_yield),
                    'type': 'forecast'
                })
        
        return {
            "status": "success",
            "data": results,
            "summary": {
                "total_years": len(results),
                "historical_years": len([r for r in results if r['type'] == 'historical']),
                "forecast_years": len([r for r in results if r['type'] == 'forecast'])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting yields: {str(e)}")

@app.post("/update-climate-forecast")
async def update_climate_forecast():
    """
    Second endpoint: Update forecasted climate data using the 6 Prophet climate models
    """
    try:
        # Load historical data
        historical_data, current_forecasted = load_data()
        if historical_data is None:
            raise HTTPException(status_code=500, detail="Failed to load historical data")
        
        # Prepare data for climate prediction
        # Use the most recent data as the basis for prediction
        last_year = historical_data['Year'].max()
        
        # Generate future dates for next 2 years (growing season only)
        future_dates = []
        for year in range(last_year + 1, last_year + 3):  # Next 2 years
            for month in GROWING_SEASON_MONTHS:
                future_date = pd.Timestamp(f"{year}-{month:02d}-01")
                future_dates.append(future_date)
        
        # Create future dataframe for predictions with growing season indicator
        future_df = pd.DataFrame({'ds': future_dates})
        future_df = add_growing_season_indicator(future_df)
        
        # Predict each climate variable
        updated_climate_data = []
        
        for idx, row in future_df.iterrows():
            climate_prediction = {
                'datetime': row['ds'],
                'year': row['ds'].year,
                'month': row['ds'].month
            }
            
            # Predict each climate feature
            for feature in CLIMATE_FEATURES:
                if feature in models:
                    try:
                        # Create a single-row dataframe for prediction with growing season
                        single_future = pd.DataFrame({
                            'ds': [row['ds']],
                            'growing_season': [row['growing_season']]
                        })
                        
                        prediction = models[feature].predict(single_future)
                        predicted_value = float(prediction['yhat'].iloc[0])
                        
                        # Apply constraints based on historical data
                        constrained_value = validate_and_constrain_predictions(
                            np.array([predicted_value]), 
                            feature, 
                            historical_data
                        )[0]
                        
                        climate_prediction[feature] = float(constrained_value)
                        
                    except Exception as model_error:
                        print(f"Error predicting {feature}: {model_error}")
                        # Try without growing_season column as fallback
                        try:
                            single_future_simple = pd.DataFrame({'ds': [row['ds']]})
                            prediction = models[feature].predict(single_future_simple)
                            predicted_value = float(prediction['yhat'].iloc[0])
                            
                            # Apply constraints
                            constrained_value = validate_and_constrain_predictions(
                                np.array([predicted_value]), 
                                feature, 
                                historical_data
                            )[0]
                            
                            climate_prediction[feature] = float(constrained_value)
                        except Exception as fallback_error:
                            print(f"Fallback failed for {feature}: {fallback_error}")
                            # Use historical average as last resort
                            historical_avg = historical_data[feature].mean()
                            climate_prediction[feature] = float(historical_avg)
                else:
                    # Fallback: use historical average if model not available
                    historical_avg = historical_data[feature].mean()
                    climate_prediction[feature] = float(historical_avg)
            
            updated_climate_data.append(climate_prediction)
        
        # Save updated forecast to CSV
        updated_df = pd.DataFrame(updated_climate_data)
        forecasted_path = os.path.join(DATA_DIR, "forecasted_climate_data_growing_season.csv")
        updated_df.to_csv(forecasted_path, index=False)
        
        # Calculate summary statistics
        summary_stats = {}
        for feature in CLIMATE_FEATURES:
            if feature in updated_df.columns:
                summary_stats[feature] = {
                    'min': float(updated_df[feature].min()),
                    'max': float(updated_df[feature].max()),
                    'mean': float(updated_df[feature].mean()),
                    'std': float(updated_df[feature].std())
                }
        
        return {
            "status": "success",
            "message": "Climate forecast updated successfully with improved Prophet models",
            "data": updated_climate_data,
            "file_updated": forecasted_path,
            "records_updated": len(updated_climate_data),
            "summary_statistics": summary_stats,
            "model_info": {
                "models_used": list(models.keys()),
                "constraints_applied": True,
                "growing_season_indicator": True
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating climate forecast: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if models are loaded
        models_loaded = len(models) > 0
        
        # Check if data files exist
        historical_exists = os.path.exists(os.path.join(DATA_DIR, "yala_season_climate_yield_data.csv"))
        forecast_exists = os.path.exists(os.path.join(DATA_DIR, "forecasted_climate_data_growing_season.csv"))
        
        return {
            "status": "healthy",
            "models_loaded": models_loaded,
            "models_count": len(models),
            "historical_data_exists": historical_exists,
            "forecast_data_exists": forecast_exists,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/model-info")
async def get_model_info():
    """Get information about loaded models"""
    try:
        model_info = {}
        for model_name, model in models.items():
            model_info[model_name] = {
                "type": str(type(model).__name__),
                "loaded": True,
                "has_growing_season": hasattr(model, 'extra_regressors') and 'growing_season' in str(model.extra_regressors) if hasattr(model, 'extra_regressors') else False
            }
        
        return {
            "status": "success",
            "models": model_info,
            "total_models": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.get("/validate-predictions")
async def validate_predictions():
    """Validate that predictions are within reasonable bounds"""
    try:
        historical_data, forecasted_data = load_data()
        if historical_data is None or forecasted_data is None:
            raise HTTPException(status_code=500, detail="Failed to load data for validation")
        
        validation_results = {}
        
        for feature in CLIMATE_FEATURES:
            if feature in forecasted_data.columns:
                hist_stats = historical_data[feature].describe()
                forecast_stats = forecasted_data[feature].describe()
                
                validation_results[feature] = {
                    "historical": {
                        "min": float(hist_stats['min']),
                        "max": float(hist_stats['max']),
                        "mean": float(hist_stats['mean']),
                        "std": float(hist_stats['std'])
                    },
                    "forecasted": {
                        "min": float(forecast_stats['min']),
                        "max": float(forecast_stats['max']),
                        "mean": float(forecast_stats['mean']),
                        "std": float(forecast_stats['std'])
                    },
                    "within_bounds": bool(
                        forecast_stats['min'] >= (hist_stats['min'] - 2 * hist_stats['std']) and
                        forecast_stats['max'] <= (hist_stats['max'] + 2 * hist_stats['std'])
                    )
                }
        
        return {
            "status": "success",
            "validation_results": validation_results,
            "overall_valid": all(result["within_bounds"] for result in validation_results.values())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating predictions: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)