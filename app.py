from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import get_random_sample_and_predict

app = FastAPI()

actual_yield_data = {
    2004: 10.5,
    2005: 9.0,
    2006: 11.75,
    2007: 9.0,
    2008: 11.5,
    2009: 17.0,
    2010: 13.0,
    2011: 13.0,
    2012: 9.5,
    2013: 14.0,
    2014: 15.5,
    2015: 11.5,
    2016: 13.5,
    2017: 13.25,
    2018: 14.25,
    2019: 12.5,
    2020: 12.25,
    2021: 10.5,
    2022: 10.25,
    2023: 9.5
}

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/predict/demo/ann")
def predict_demo_ann():
    """
    Predicts the yield using a test records ANN model.
    """
    results = get_random_sample_and_predict()

    inputRecordYear = results["year"]

    frontEndData = []

    for i in range(2004, 2024):
        if i == inputRecordYear:
            break
        frontEndData.append({
            "year": i,
            "actual_yield": actual_yield_data[i]
        })

    return {
        "input": results["input"],
        "predicted_yield": results["predicted_yield"],
        "actual_yield": results["actual_yield"],
        "current year": results["year"],
        "years": frontEndData

    }

    