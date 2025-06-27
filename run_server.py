#!/usr/bin/env python3
"""
Startup script for the Onion Yield Prediction API
"""
import uvicorn
import os
import sys

def check_files():
    """Check if required files exist"""
    required_files = [
        "models/prophet_yield_model.pkl",
        "models/tempmin_prophet_model.pkl",
        "models/tempmax_prophet_model.pkl",
        "models/temp_prophet_model.pkl",
        "models/sunshine_hours_prophet_model.pkl",
        "models/rainfall_prophet_model.pkl",
        "models/humidity_prophet_model.pkl",
        "data/yala_season_climate_yield_data.csv",
        "data/forecasted_climate_data_growing_season.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all model and data files are in the correct directories.")
        return False
    
    print("✅ All required files found!")
    return True

def main():
    print("🧅 Starting Onion Yield Prediction API Server...")
    print("=" * 50)
    
    # Check if files exist
    if not check_files():
        sys.exit(1)
    
    print("🚀 Starting server on http://localhost:8000")
    print("📖 API Documentation available at http://localhost:8000/docs")
    print("🔄 Interactive API at http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()