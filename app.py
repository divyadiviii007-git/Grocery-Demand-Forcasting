import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Grocery Demand Forecasting",
    page_icon="🥦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #15803D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .forecast-card {
        background: linear-gradient(135deg, #16A34A 0%, #15803D 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 10px 15px -3px rgba(22, 163, 74, 0.3);
    }
    .forecast-units {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.4rem 0;
    }
    .info-card {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🥦 Grocery Retail Demand Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict inventory requirements and daily grocery sales using <b>RandomForestRegressor</b> with seasonality, pricing, and promotional factors.</div>', unsafe_allow_html=True)

model_path = BASE_DIR / "models" / "grocery_demand_model.pkl"
csv_path = BASE_DIR / "data" / "grocery_sales.csv"
chart_path = BASE_DIR / "outputs" / "feature_importance.png"

if not model_path.exists():
    st.error(f"Model file not found at `{model_path}`! Please run 'python train_model.py' first.")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(str(model_path))

model = get_model()

tab1, tab2, tab3 = st.tabs(["🔮 Demand Forecaster & Inventory Planner", "📈 Feature Importance & Trends", "📋 Historical Sales Data"])

with tab1:
    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("Forecast Parameters")
        
        scenario = st.selectbox(
            "⚡ Quick Retail Scenario Preset",
            ["Custom Settings", "🔥 Weekend Promotional Surge", "🎉 Holiday Sale Event", "🛒 Midweek Regular Trading", "📉 Low Demand Slow Day"]
        )
        
        # Determine defaults based on scenario
        if scenario == "🔥 Weekend Promotional Surge":
            def_dow, def_price, def_promo, def_hol, def_prev, def_roll = 5, 89.0, 1, 0, 150, 135
        elif scenario == "🎉 Holiday Sale Event":
            def_dow, def_price, def_promo, def_hol, def_prev, def_roll = 6, 92.0, 1, 1, 160, 140
        elif scenario == "🛒 Midweek Regular Trading":
            def_dow, def_price, def_promo, def_hol, def_prev, def_roll = 2, 100.0, 0, 0, 85, 90
        elif scenario == "📉 Low Demand Slow Day":
            def_dow, def_price, def_promo, def_hol, def_prev, def_roll = 0, 105.0, 0, 0, 60, 75
        else:
            def_dow, def_price, def_promo, def_hol, def_prev, def_roll = 5, 95.0, 1, 0, 120, 110
        
        days_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        
        with st.form("demand_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                day_selected = st.selectbox(
                    "Day of the Week",
                    options=list(days_map.keys()),
                    index=def_dow,
                    format_func=lambda x: f"{days_map[x]} (Day {x})"
                )
                price = st.slider("Unit Selling Price (₹)", 40.0, 200.0, float(def_price), step=1.0)
                previous_sales = st.slider("Previous Day Sales (Units)", 10, 400, int(def_prev), step=5)
            with col_b:
                rolling_sales = st.slider("7-Day Rolling Average Sales", 10, 400, int(def_roll), step=5)
                promotion = st.radio("Active Promotional Discount?", [1, 0], index=0 if def_promo == 1 else 1, format_func=lambda x: "Yes (Promoted)" if x == 1 else "No Promo")
                holiday = st.radio("Calendar / Festival Holiday?", [0, 1], index=0 if def_hol == 0 else 1, format_func=lambda x: "No (Standard Day)" if x == 0 else "Yes (Holiday)")
                
            submit_btn = st.form_submit_button("🚀 Generate Demand Forecast", use_container_width=True)
            
    with col_result:
        st.subheader("Inventory Forecast")
        if submit_btn:
            input_df = pd.DataFrame([{
                "day_of_week": day_selected,
                "price": price,
                "promotion": promotion,
                "holiday": holiday,
                "previous_day_sales": previous_sales,
                "rolling_7_day_sales": rolling_sales
            }])
            
            raw_forecast = model.predict(input_df)[0]
            forecast_units = max(0, round(raw_forecast))
            expected_revenue = forecast_units * price
            safety_buffer = round(forecast_units * 0.15)  # 15% safety stock
            recommended_stock = forecast_units + safety_buffer
            
            st.markdown(f"""
            <div class="forecast-card">
                <div style="font-size: 1rem; opacity: 0.9;">Projected Grocery Demand</div>
                <div class="forecast-units">{forecast_units:,} Units</div>
                <div style="font-size: 1.1rem; opacity: 0.95;">Estimated Daily Revenue: ₹{expected_revenue:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Safety Buffer (+15%)", f"{safety_buffer} units")
            col_m2.metric("Recommended Total Stock", f"{recommended_stock} units")
            
            st.markdown("### 📦 Supply Chain Recommendations")
            if promotion == 1:
                st.info("🏷️ **Active Promotion**: Promotional elasticity increases velocity. Ensure end-cap shelf stock is replenished by morning.")
            if holiday == 1:
                st.warning("🎉 **Holiday Surge**: Expect higher footfall peak between 11 AM - 3 PM. Staff front counters accordingly.")
            if forecast_units > rolling_sales:
                growth_pct = ((forecast_units - rolling_sales) / max(rolling_sales, 1)) * 100
                st.success(f"📈 Demand is projected to be **{growth_pct:.1f}% above** the 7-day trailing average.")
                
            with st.expander("🔍 Model Input Vector"):
                st.json(input_df.to_dict(orient="records")[0])
        else:
            st.info("👈 Set demand drivers and click **'Generate Demand Forecast'**.")

with tab2:
    st.subheader("Random Forest Feature Importance")
    if chart_path.exists():
        st.image(str(chart_path), caption="Feature Importance for Grocery Demand Forecasting", use_container_width=True)
    else:
        st.info("Feature importance plot will appear after running train_model.py")

with tab3:
    st.subheader("Training Records (grocery_sales.csv)")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Historical Records", f"{len(df):,}")
        col2.metric("Average Daily Demand", f"{df['demand'].mean():.1f} units")
        col3.metric("Peak Single-Day Demand", f"{df['demand'].max():.0f} units")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Dataset not found.")
