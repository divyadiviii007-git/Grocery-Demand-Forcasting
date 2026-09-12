# Grocery-Demand-Forcasting
Grocery Demand Forecasting is a machine learning project designed to predict the expected demand for grocery products based on historical sales and important business factors.
# Grocery Demand Forecasting Using Scikit-learn

## Overview

Grocery Demand Forecasting is a machine learning project designed to predict the expected demand for grocery products based on historical sales and important business factors.

The project uses a **Random Forest Regression** model to forecast daily grocery demand. The model considers factors such as the day of the week, product price, promotional activity, holidays, previous-day sales, and the 7-day rolling average of sales.

The prediction can help grocery retailers improve **inventory planning, stock management, sales forecasting, and supply-chain decisions**.

## Key Features

* Predicts daily grocery demand in units.
* Uses a Random Forest Regression machine learning algorithm.
* Considers price, promotions, holidays, and sales history.
* Provides a 15% safety-stock recommendation.
* Estimates expected daily revenue.
* Displays feature importance.
* Provides historical sales data through the web interface.
* Includes predefined retail scenarios such as:

  * Weekend Promotional Surge
  * Holiday Sale Event
  * Midweek Regular Trading
  * Low Demand Slow Day
* Provides an interactive **Streamlit dashboard**.

## Machine Learning Model

The project uses:

**Algorithm:** RandomForestRegressor

### Input Features

1. `day_of_week` – Day of the week.
2. `price` – Unit selling price.
3. `promotion` – Whether a promotion is active.
4. `holiday` – Whether the day is a holiday.
5. `previous_day_sales` – Sales from the previous day.
6. `rolling_7_day_sales` – Average sales over the previous seven days.

### Target

`demand` – Predicted grocery demand in units.

## How It Works

The historical grocery sales dataset is loaded from `grocery_sales.csv`. The data is divided into training and testing sets.

A Random Forest Regression model is trained using 80% of the data and evaluated using the remaining 20%.

The model calculates:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

The trained model is saved as a `.pkl` file and can then be used to generate predictions for future demand.

## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib
* Streamlit

## Applications

This project can be useful for:

* Grocery stores
* Supermarkets
* Inventory management
* Retail demand planning
* Supply-chain management
* Stock replenishment
* Sales forecasting

## Future Improvements

The project can be further enhanced by:

* Using real-world retail datasets.
* Adding product/category-level forecasting.
* Incorporating weather and seasonal information.
* Adding monthly and yearly trends.
* Comparing Random Forest with XGBoost, Gradient Boosting, or other models.
* Adding automated model retraining.
* Connecting the application to a live database.
* Adding interactive sales and demand charts.
