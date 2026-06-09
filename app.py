import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

st.set_page_config(
    page_title="Bike Demand Forecast",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚲 Bike Demand Forecasting")

st.info("""
    ⚠️ **Prediction Disclaimer:** 
    - Does not account for special events or infrastructure changes
    - For planning purposes only — not real-time operations"""
    )

st.divider()

# Load model
@st.cache_resource
def load_model():
    possible_paths = [
        "lightgbm_model.pkl",
        "notebook/lightgbm_model.pkl",
        "../lightgbm_model.pkl",
        "../notebook/lightgbm_model.pkl"
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            return joblib.load(model_path)
    
    st.error("Model file not found. Please run the notebook first to train and save the model.")
    return None

model = load_model()

# Season mapping
def get_season_from_month(month):
    if month in [12, 1, 2]:
        return 1, "Winter"
    elif month in [3, 4, 5]:
        return 2, "Spring"
    elif month in [6, 7, 8]:
        return 3, "Summer"
    else:
        return 4, "Fall"

def get_default_temp_hum_wind(season_name, weathersit):
    if season_name == "Summer":
        temp_default = 0.75
        hum_default = 0.55
    elif season_name == "Spring":
        temp_default = 0.60
        hum_default = 0.60
    elif season_name == "Fall":
        temp_default = 0.50
        hum_default = 0.65
    else:
        temp_default = 0.35
        hum_default = 0.70
    
    if weathersit == 2:
        temp_default = temp_default - 0.05
        hum_default = min(0.95, hum_default + 0.10)
    elif weathersit == 3:
        temp_default = temp_default - 0.10
        hum_default = min(0.95, hum_default + 0.20)
    elif weathersit == 4:
        temp_default = temp_default - 0.15
        hum_default = 0.95
    
    wind_default = 0.20
    
    return temp_default, hum_default, wind_default

st.sidebar.header("📊 Input Parameters")

st.sidebar.info("""
    📅 **Note:** The model was trained on data from **2011 to June 2012**.
    Predictions are for **July 2012 - December 2012** (test period).
""")

# Temporal features
st.sidebar.subheader("⏰ Temporal Features")

hour = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 12)

# Month selection
month_names = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

month = st.sidebar.selectbox("Month", options=list(month_names.keys()), format_func=lambda x: month_names[x])

# Auto-calculate season
season_value, season_name = get_season_from_month(month)
st.sidebar.markdown(f"**🍂 Season:** {season_name} (auto-detected)")

# Year - Only 2012
year = st.sidebar.selectbox("Year", [2012], index=0)
st.sidebar.caption("Test period: July-December 2012")

# Day of week
weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday = st.sidebar.selectbox("Day of Week", options=list(range(7)), format_func=lambda x: weekday_names[x])

# Working day auto-detection
def get_workingday_from_weekday(weekday, is_holiday):
    is_weekend = (weekday == 5 or weekday == 6)
    if is_weekend or is_holiday:
        return 0
    else:
        return 1

auto_workingday = get_workingday_from_weekday(weekday, False)

if weekday == 5 or weekday == 6:
    st.sidebar.markdown(f"**💼 Working Day:** No (Weekend - auto-detected)")
else:
    st.sidebar.markdown(f"**💼 Working Day:** Yes (auto-detected)")

holiday = st.sidebar.checkbox("Is Holiday?")
if holiday:
    auto_workingday = 0
    st.sidebar.markdown(f"**💼 Working Day:** No (Holiday - overrides)")

st.sidebar.divider()

# Weather features
st.sidebar.subheader("🌤️ Weather Features")

weathersit_names = {
    1: "Clear/Few clouds",
    2: "Mist/Cloudy",
    3: "Light rain/snow",
    4: "Heavy rain/snow"
}
weathersit = st.sidebar.selectbox("Weather Situation", options=list(weathersit_names.keys()), format_func=lambda x: weathersit_names[x])

default_temp, default_hum, default_wind = get_default_temp_hum_wind(season_name, weathersit)

temp = st.sidebar.slider("Temperature (normalized 0-1)", 0.0, 1.0, default_temp, step=0.01)
hum = st.sidebar.slider("Humidity (normalized 0-1)", 0.0, 1.0, default_hum, step=0.01)
windspeed = st.sidebar.slider("Wind Speed (normalized 0-1)", 0.0, 1.0, default_wind, step=0.01)

st.sidebar.divider()

def create_cyclical_features(hour, month, weekday):
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    dow_sin = np.sin(2 * np.pi * weekday / 7)
    dow_cos = np.cos(2 * np.pi * weekday / 7)
    
    return hour_sin, hour_cos, month_sin, month_cos, dow_sin, dow_cos

def get_lag_features(hour, recent_avg=150):
    if hour in [7, 8, 9, 16, 17, 18]:
        cnt_lag_1 = 450
        cnt_lag_2 = 420
        cnt_lag_3 = 400
        cnt_lag_24 = 430
        cnt_lag_168 = 410
        cnt_rolling_24 = 420
        cnt_rolling_168 = 400
    elif hour in [22, 23, 0, 1, 2, 3, 4, 5]:
        cnt_lag_1 = 50
        cnt_lag_2 = 45
        cnt_lag_3 = 40
        cnt_lag_24 = 60
        cnt_lag_168 = 55
        cnt_rolling_24 = 50
        cnt_rolling_168 = 45
    else:
        cnt_lag_1 = 200
        cnt_lag_2 = 190
        cnt_lag_3 = 180
        cnt_lag_24 = 200
        cnt_lag_168 = 190
        cnt_rolling_24 = 195
        cnt_rolling_168 = 185
    
    return cnt_lag_1, cnt_lag_2, cnt_lag_3, cnt_lag_24, cnt_lag_168, cnt_rolling_24, cnt_rolling_168

# Main content area
st.subheader("🔮 Make a Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Hour", f"{hour}:00")
    st.metric("Month", month_names[month])
    st.metric("Season", season_name)

with col2:
    st.metric("Temperature", f"{temp:.2f}")
    st.metric("Humidity", f"{hum:.2f}")

with col3:
    st.metric("Wind Speed", f"{windspeed:.2f}")
    st.metric("Weather", weathersit_names[weathersit])

# Display working day status
if holiday:
    st.info("📅 **Holiday Mode:** Working Day = No (Holiday override)")
elif weekday == 5 or weekday == 6:
    st.info("📅 **Weekend Mode:** Working Day = No (Auto-detected)")
else:
    st.success("📅 **Weekday Mode:** Working Day = Yes (Auto-detected)")

st.caption("ℹ️ **Model Test Period:** July 2012 - December 2012")

if st.button("🚲 Predict Bike Rentals", type="primary", use_container_width=True):
    if model is not None:
        hour_sin, hour_cos, month_sin, month_cos, dow_sin, dow_cos = create_cyclical_features(
            hour, month, weekday
        )
        
        cnt_lag_1, cnt_lag_2, cnt_lag_3, cnt_lag_24, cnt_lag_168, cnt_rolling_24, cnt_rolling_168 = get_lag_features(hour)
        
        yr = 1 if year == 2012 else 0
        season = season_value
        workingday = auto_workingday
        
        features = np.array([[
            season, yr, 1 if holiday else 0, workingday,
            weathersit, temp, hum, windspeed, hour, month, weekday,
            hour_sin, hour_cos, month_sin, month_cos, dow_sin, dow_cos,
            cnt_lag_1, cnt_lag_2, cnt_lag_3,
            cnt_lag_24, cnt_lag_168, cnt_rolling_24, cnt_rolling_168
        ]])
        
        prediction = model.predict(features)[0]
        prediction = max(0, int(round(prediction)))
        
        st.divider()
        st.subheader(f"📊 Historical Data for {weekday_names[weekday]}s in {month_names[month]} {year}")
        
        csv_paths = [
            "test_predictions.csv",
            "notebook/test_predictions.csv",
            "../test_predictions.csv",
            "../notebook/test_predictions.csv"
        ]
        
        csv_found = False
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                df_predictions = pd.read_csv(csv_path)
                csv_found = True
                break
        
        if csv_found:
            if 'Date' in df_predictions.columns:
                df_predictions['Date_dt'] = pd.to_datetime(df_predictions['Date'])
                
                similar_predictions = df_predictions[
                    (df_predictions['Date_dt'].dt.month == month) &
                    (df_predictions['Date_dt'].dt.year == year) &
                    (df_predictions['Hour'] == hour) &
                    (df_predictions['Date_dt'].dt.dayofweek == weekday)
                ].copy()
                
                if len(similar_predictions) > 0:
                    similar_predictions['Day'] = similar_predictions['Date_dt'].dt.day
                    
                    st.markdown(f"### 📅 All {weekday_names[weekday]}s in {month_names[month]} {year} at {hour}:00")
                    
                    comparison_df = similar_predictions[['Date', 'Actual', 'Predicted', 'Error']].copy()
                    comparison_df = comparison_df.reset_index(drop=True)
                    comparison_df.index = comparison_df.index + 1
                    comparison_df['Date'] = pd.to_datetime(comparison_df['Date']).dt.strftime('%Y-%m-%d')
                    comparison_df['Error'] = comparison_df['Error'].round(1)
                    
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    avg_actual = similar_predictions['Actual'].mean()
                    avg_predicted = similar_predictions['Predicted'].mean()
                    avg_error = similar_predictions['Error'].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Average Actual", f"{avg_actual:.0f} bikes")
                    with col2:
                        st.metric("🤖 Average Predicted", f"{avg_predicted:.0f} bikes")
                    with col3:
                        st.metric("📈 Average Error", f"{avg_error:.1f} bikes")
                    
                    st.subheader(f"📈 Actual vs Predicted for {weekday_names[weekday]}s in {month_names[month]} {year} at {hour}:00")
                    
                    chart_data = pd.DataFrame({
                        'Day': similar_predictions['Day'],
                        'Actual': similar_predictions['Actual'],
                        'Predicted': similar_predictions['Predicted']
                    }).sort_values('Day')
                    
                    st.line_chart(chart_data.set_index('Day'), use_container_width=True)
                    
                    diff_from_avg = abs(prediction - avg_actual)
                    if diff_from_avg < 20:
                        st.success(f"✅ Your prediction ({prediction} bikes) is very close to historical average ({avg_actual:.0f} bikes)!")
                    elif diff_from_avg < 50:
                        st.info(f"ℹ️ Your prediction ({prediction} bikes) is reasonably close to historical average ({avg_actual:.0f} bikes)")
                else:
                    st.info(f"No historical data found for {weekday_names[weekday]}s in {month_names[month]} {year} at {hour}:00")
        else:
            st.warning("test_predictions.csv not found. Run the notebook first to generate predictions.")
        
        st.caption(f"Prediction based on: {hour}:00, {weekday_names[weekday]}, {month_names[month]}, {weathersit_names[weathersit]}")
    else:
        st.error("Model not loaded. Please ensure 'lightgbm_model.pkl' exists.")