# 📊 Dashboard Summary: Engine Maintenance Predictor

## 📌 Project Purpose
This dashboard uses real-time engine parameters to:
- Predict **how many hours are left before the next maintenance is needed**
- Flag whether the engine is operating **within safe limits**
- Show **historical trends** in key engine metrics

---

## 🔍 Key Insights

- **Maintenance Prediction**: The model estimates time to next maintenance based on parameters like oil pressure, temperature, RPM, etc.
- **Safety Monitoring**: The system checks if vibration, fuel usage, temperature, and pressure are within acceptable thresholds.
- **Historical Trends**: Visualizations help spot anomalies or degradation over time in:
  - Oil Pressure
  - Engine Temperature
  - Fuel Consumption
  - Engine Load
  - RPM

---

## ⚠️ Maintenance Alerts
- If predicted maintenance time is **less than 5 hours**, the system displays a **critical alert**.
- If any parameters fall outside safe ranges, an **"Unsafe"** status is flagged.

---

## 📈 Visualizations
- **Moving Average** of temperature and oil pressure
- **Line Charts** of key parameters over engine hours
- **Interactive Graphs** built with Plotly for better interpretation

---

## 🧠 ML Model Used
- **Random Forest Regressor** trained to predict `Maintenance Due Next`
- Features used:
  - `Hours`, `Vibration Level`, `Temp`, `Oil pressure`, `Fuel`, `Load`, `RPM`
- Model Evaluation:
  - **MAE**: Mean absolute error
  - **RMSE**: Root mean squared error
  - **R² Score**: Variance explained

---

## ✅ Project Status
- ✅ Dashboard tested and working
- ✅ Model trained and saved
- ✅ Ready for GitHub deployment and Streamlit Cloud hosting

---

## 📁 Data Source
- Data collected from MT Zaria engine logs
- Cleaned dataset: `mt_zaria_engine_log_cleaned2.csv`

