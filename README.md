# 🚲 Bike Demand Forecasting

🔗 Live Demo: https://bikedemand.onrender.com

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3%2B-green)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-red)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📌 Overview

An end-to-end machine learning project that predicts **hourly bike rental demand** using weather and temporal features. Built with proper time series validation (no data leakage!) and achieving **91.62% explained variance** on future test data.

**Key Achievement:** R² = 0.9162 on completely unseen future data (July–December 2012)

---

```

## 🏗️ System Architecture

┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                  1. DATA PIPELINE                    │   │
│   │  • Raw Data (CSV Files) • 17,379 hourly records     │   │
│   └─────────────────────────┬───────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                2. FEATURE ENGINEERING                │   │
│   │  • Sin/Cos Encoding • Lag Features (24h, 168h)      │   │
│   │  • Rolling Averages • Weather Features              │   │
│   └─────────────────────────┬───────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                   3. MODEL TRAINING                  │   │
│   │  • LightGBM (Best: 63.62 RMSE) • XGBoost            │   │
│   │  • Random Forest • Gradient Boosting                │   │
│   │  • TimeSeriesSplit (5 folds)                        │   │
│   └─────────────────────────┬───────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                  4. MODEL EVALUATION                 │   │
│   │  • SHAP Analysis • Weather Impact                   │   │
│   │  • Error by Hour • Business Metrics                 │   │
│   └─────────────────────────┬───────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                    5. DEPLOYMENT                     │   │
│   │  • Streamlit Web App • Real-time Predictions        │   │
│   │  • Interactive Interface • API Ready                │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow Diagram

```

┌─────────────────────────────────────────────────────────────┐
│                     DATA FLOW DIAGRAM                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                  1. RAW DATA                         │   │
│   │         hour.csv • day.csv (UCI Dataset)            │   │
│   └─────────────────────────┬───────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                 2. DATA PREPROCESSING                │   │
│   │  • Handle missing values • Convert date to datetime │   │
│   │  • Remove leakage columns (casual, registered)      │   │
│   │  • Handle outliers using IQR method                 │   │
│   └─────────────────────────┬───────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                 3. FEATURE ENGINEERING               │   │
│   │  • Cyclical encoding (sin/cos for hour/month/day)   │   │
│   │  • Lag features (24h and 168h shifts)               │   │
│   │  • Rolling averages (24h and 7-day windows)         │   │
│   └─────────────────────────┬───────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                   4. MODEL TRAINING                  │   │
│   │  • Train/Test split (time-based: July 2012 cutoff)  │   │
│   │  • Train: 12,331 rows (2011 + early 2012)           │   │
│   │  • Test: 4,376 rows (July–Dec 2012)                 │   │
│   │  • TimeSeriesSplit cross-validation (5 folds)       │   │
│   │  • Hyperparameter tuning with Optuna                │   │
│   └─────────────────────────┬───────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                  5. MODEL EVALUATION                 │   │
│   │  • SHAP analysis for interpretability               │   │
│   │  • Weather impact analysis (error by condition)     │   │
│   │  • Error analysis by hour of day                    │   │
│   │  • Business metrics (MAE: 42 bikes, 16.6% error)    │   │
│   └─────────────────────────┬───────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                   6. DEPLOYMENT                      │   │
│   │  • Streamlit web application                        │   │
│   │  • Real-time predictions                            │   │
│   │  • User-friendly interface                          │   │
│   │  • CSV export functionality                         │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
---

### Detailed Component Breakdown:

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| **1. Raw Data** | UCI CSV files | Load and inspect | Clean DataFrame |
| **2. Preprocess** | Raw DataFrame | Handle missing, convert dates, remove leakage | Processed Data |
| **3. Feature Engineer** | Processed Data | Sin/Cos, Lag features, Rolling averages | Feature Matrix (21 features) |
| **4. Model Training** | Feature Matrix | TimeSeriesSplit CV, Hyperparameter tuning | Trained Models |
| **5. Evaluation** | Predictions | SHAP, Weather impact, Error analysis | Metrics & Visualizations |
| **6. Deployment** | Trained Model | Streamlit web app | Real-time predictions |

---

## 📸 Web Application Screenshots

!(screenshots/home_page.png)

!(screenshots/prediction_result.png)

!(screenshots/sample_predictions.png)

---

## 🎯 Problem Statement

Bike sharing systems generate massive amounts of spatio-temporal data. Accurately forecasting demand helps operators:
- Optimize bike redistribution across stations
- Schedule maintenance during low-demand hours
- Improve user satisfaction by reducing empty/full stations

This project predicts the **total number of bike rentals per hour** (`cnt`) using:
- Temporal features (hour, month, day of week, holiday status)
- Weather features (temperature, humidity, wind speed)
- Engineered cyclical patterns (sin/cos transformations)
- **Time series lag features** (24-hour and 168-hour lags)

---

## 📊 Dataset

**Source:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)

| Feature | Description |
|---------|-------------|
| `dteday` | Date |
| `season` | 1: winter, 2: spring, 3: summer, 4: fall |
| `yr` | 0: 2011, 1: 2012 |
| `mnth` | Month (1–12) |
| `hr` | Hour (0–23) |
| `holiday` | Whether the day is a holiday |
| `weekday` | Day of week (0–6) |
| `workingday` | 1 if neither weekend nor holiday |
| `weathersit` | 1: Clear, 2: Mist, 3: Light rain/snow, 4: Heavy rain/snow |
| `temp` | Normalized temperature |
| `atemp` | Normalized "feels like" temperature (dropped due to high correlation) |
| `hum` | Normalized humidity |
| `windspeed` | Normalized wind speed |
| `casual` | Number of casual users (⚠️ **leakage** - removed) |
| `registered` | Number of registered users (⚠️ **leakage** - removed) |
| `cnt` | **Target:** Total bike rentals |

**Dataset Size:** 17,379 hourly records (2011–2012) → 16,875 after lag features

---

## 🔧 Feature Engineering

### 1. Cyclical Encoding (sin/cos transformations)

Preserves circular relationships (e.g., hour 23 and hour 0 are close):

```python
df['hour_sin'] = np.sin(2 * π * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * π * df['hour'] / 24)
df['month_sin'] = np.sin(2 * π * df['month'] / 12)
df['month_cos'] = np.cos(2 * π * df['month'] / 12)
df['dow_sin'] = np.sin(2 * π * df['weekday'] / 7)
df['dow_cos'] = np.cos(2 * π * df['weekday'] / 7)
```

### 2. Time Series Lag Features

```python
df['cnt_lag_24'] = df['cnt'].shift(24)      # Same hour yesterday
df['cnt_lag_168'] = df['cnt'].shift(168)    # Same hour last week
df['cnt_rolling_24'] = df['cnt'].rolling(24).mean()   # 24-hour average
df['cnt_rolling_168'] = df['cnt'].rolling(168).mean() # Weekly average
```

### 3. Leakage Prevention

- `casual` and `registered` dropped (these sum to `cnt` and would cheat the model)
- `instant` and `dteday` dropped

---

## 🧠 Models Compared

| Model | RMSE | R² | MAE |
|-------|------|-----|-----|
| **LightGBM (Default)** | **63.62** | **0.9162** | **41.74** |
| LightGBM (Optimized) | 64.70 | 0.9133 | 40.76 |
| XGBoost | 67.94 | 0.9044 | 44.53 |
| Random Forest | 75.06 | 0.8833 | 49.35 |
| Gradient Boosting | 83.09 | 0.8570 | 52.89 |
| Linear Regression | 166.53 | 0.4254 | 107.25 |

**🏆 Best Model:** LightGBM (Default) with R² = 0.9162

---

## ⏱️ Time Series Validation (Critical!)

```python
# PROPER time-based split - NO SHUFFLING!
split_date = '2012-07-01'
train_mask = df['dteday'] < split_date   # 12,331 rows (2011 + early 2012)
test_mask = df['dteday'] >= split_date   # 4,376 rows (July–Dec 2012)

X_train = X[train_mask]  # Past data only
X_test  = X[test_mask]   # Future data only
```

**Why this matters:** Random shuffling would leak future information into training, making results invalid for forecasting.

---

## 📈 Model Interpretability (SHAP Analysis)

SHAP values explain which features most influence predictions:

| Feature | Impact |
|---------|--------|
| `cnt_rolling_24` (24-hour average) | Highest positive impact |
| `hour_sin` / `hour_cos` | Strong cyclical patterns |
| `temp` | Positive correlation with demand |
| `weathersit` | Negative impact during rain/snow |

---

## 🌤️ Weather Impact Analysis

| Weather Condition | Mean Error (bikes) |
|-------------------|-------------------|
| Clear (1) | 37.8 |
| Mist/Cloudy (2) | 44.5 |
| Light Rain/Snow (3) | 75.5 |
| Heavy Rain/Snow (4) | No data in test set |

**Key Insights:**
- Predictions are most accurate in clear weather (38 bikes error)
- Error increases by ~37 bikes during light rain/snow

---

## 📊 Key Insights from EDA

| Finding | Implication |
|---------|-------------|
| Peak commute: 5 PM (525 avg rentals) | Staff more bikes during evening rush |
| Deepest night: 3–4 AM (5–8 rentals) | Schedule maintenance overnight |
| Weekend peak: 1 PM (373 avg rentals) | Casual users ride midday |
| Temperature correlation: 0.24 (night) → 0.53 (evening) | Weather impacts evenings most |
| Growth: +75–134 bikes/month from 2011 to 2012 | Demand increased significantly |

### Prediction Error by Hour

| Best Hours (Lowest Error) | Worst Hours (Highest Error) |
|---------------------------|----------------------------|
| 4 AM: 3.2 bikes | 5 PM: 86.8 bikes |
| 3 AM: 5.1 bikes | 6 PM: 75.0 bikes |
| 2 AM: 9.7 bikes | 8 AM: 72.9 bikes |
| 5 AM: 9.7 bikes | 4 PM: 62.5 bikes |
| 1 AM: 13.1 bikes | 3 PM: 61.4 bikes |

---

## 💼 Business Impact

| Metric | Value |
|--------|-------|
| Mean Absolute Error | 42 bikes |
| Percentage Error | 16.6% |
| Estimated Annual Revenue | **$11,129,174** |

---

## 📁 Project Structure

```
BikeDemand/
│
├── BikeDemand.ipynb                 # Main analysis notebook
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore file
├── README.md                        # Project documentation
│
├── lightgbm_model.pkl               # Saved trained model
├── test_predictions.csv             # All predictions
├── feature_importance.csv           # Feature importance rankings
├── final_metrics.txt                # Model performance metrics
├── business_metrics.txt             # Business impact metrics
│
├── shap_summary.png                 # SHAP dot plot
├── shap_importance.png              # SHAP bar plot
├── weather_boxplot.png              # Weather error distribution
├── weather_temp_error.png           # Temperature vs error plot
├── weather_impact_analysis.png      # Combined weather analysis
│
├── dataset/
    ├── hour.csv                     # Hourly bike data (UCI)
    └── day.csv                      # Daily bike data (UCI)

```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/boughanmiyoussef/BikeDemand.git
cd BikeDemand
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter notebook
```bash
jupyter notebook BikeDemand.ipynb
```

### 5. Run the Streamlit web app
```bash
streamlit run app.py
```

---

## 🌐 Web Application (Streamlit)

The project includes an `app.py` file that creates an interactive web application:

### Features:
- Input hour, temperature, humidity, wind speed, and weather condition
- Real-time prediction using the trained LightGBM model
- Visual display of predicted bike rentals
- Easy-to-use interface for non-technical users

### Sample usage:
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | LightGBM, XGBoost, Random Forest, Gradient Boosting |
| **Hyperparameter Tuning** | Optuna |
| **Model Interpretability** | SHAP |
| **Time Series Validation** | TimeSeriesSplit |
| **Web Deployment** | Streamlit |

---

## 📝 Future Improvements

- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Incorporate weather forecast API for real-time predictions
- [ ] Experiment with Prophet or LSTM for comparison
- [ ] Add prediction intervals for uncertainty quantification
- [ ] Create Docker container for easy deployment
- [ ] Add CI/CD pipeline for automated testing

---

## 📝 License

This project is licensed under the MIT License.

---

## 👤 Author

**Youssef Boughanmi**

- 📧 yussefboughanmy@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/youssef-boughanmi-4990222a0)
- 💻 [GitHub](https://github.com/boughanmiyoussef)

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Bike Sharing dataset
- LightGBM, XGBoost, and Scikit-learn teams
- Optuna developers for automated hyperparameter optimization
- SHAP team for model interpretability tools

---

## ⭐ If you found this useful, please star the repository!

[![GitHub stars](https://img.shields.io/github/stars/boughanmiyoussef/BikeDemand?style=social)](https://github.com/boughanmiyoussef/BikeDemand)