import os
import sys
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "grocery_demand_model.pkl"
model = joblib.load(MODEL_PATH)

future_day = pd.DataFrame([{
    "day_of_week": 5,
    "price": 95,
    "promotion": 1,
    "holiday": 0,
    "previous_day_sales": 120,
    "rolling_7_day_sales": 110
}])

forecast = model.predict(future_day)[0]
print(f"Forecasted grocery demand: {forecast:.0f} units")
