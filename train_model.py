import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/grocery_sales.csv"
MODEL_PATH = "models/grocery_demand_model.pkl"

df = pd.read_csv(DATA_PATH)

features = [
    "day_of_week",
    "price",
    "promotion",
    "holiday",
    "previous_day_sales",
    "rolling_7_day_sales"
]
target = "demand"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("Grocery Demand Forecasting Results")
print("-" * 42)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

importance = pd.Series(
    model.feature_importances_, index=features
).sort_values()

importance.plot(kind="barh", title="Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/feature_importance.png")
plt.close()
print("Chart saved to: outputs/feature_importance.png")
