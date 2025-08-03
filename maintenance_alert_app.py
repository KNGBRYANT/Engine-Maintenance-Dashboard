import streamlit as st
import pandas as pd
import joblib  # to load your trained model
from datetime import datetime

# Load data
df = pd.read_csv("mt_zaria_engine_log_cleaned2.csv")


# Load model
model = joblib.load("C:/Users/Mayo/Documents/All Projects/ML_project1/regression_model.pkl")
  # we'll save this model shortly

# User input (simulating real-time values)
st.title("🚨 Engine Maintenance Dashboard")

st.sidebar.header("Input Current Engine Parameters")
hours = st.sidebar.slider("Hours", 0, 10000, 500)
vibration = st.sidebar.slider("Vibration Level", 0.0, 5.0, 0.5)
temp = st.sidebar.slider("Temperature (°C)", 50, 120, 85)
oil = st.sidebar.slider("Oil Pressure (bar)", 0.0, 10.0, 3.0)
fuel = st.sidebar.slider("Fuel Consumption (lph)", 0, 50, 20)
load = st.sidebar.slider("Engine Load (%)", 0, 100, 60)
rpm = st.sidebar.slider("RPM", 200, 2000, 900)

# Create a DataFrame
input_data = pd.DataFrame([{
    'Hours': hours,
    'Vibration Level': vibration,
    'Temp': temp,
    'Oil pressure': oil,
    'Fuel': fuel,
    'Load': load,
    'RPM': rpm
}])

# Predict maintenance time left
pred_hours = model.predict(input_data)[0]

# Safety check
safe_ranges = {
    'Vibration Level': (0.1, 1.5),
    'Temp': (75, 95),
    'Oil pressure': (2.0, 5.0),
    'Fuel': (10, 30),
    'Load': (30, 90),
    'RPM': (300, 1200)
}

def check_safety(row):
    for param, (min_val, max_val) in safe_ranges.items():
        if not (min_val <= row[param] <= max_val):
            return "Unsafe"
    return "Safe"

status = check_safety(input_data.iloc[0])

# Display results
st.subheader("🔍 Current Engine Status")
st.write(f"**Status:** {status}")
st.write(f"**Predicted Hours Until Maintenance:** {pred_hours:.2f} hours")

if pred_hours < 5 or status == "Unsafe":
    st.error("⚠️ Immediate maintenance action recommended!")
else:
    st.success("✅ Engine is operating within safe parameters.")


# Section: Data Overview & Trends
st.markdown("---")
st.subheader("📊 Data Overview & Trends")

# Show the first few rows of the dataset
if st.checkbox("Show raw data"):
    num_rows = st.slider("Number of rows to display", 5, 50, 10)
    df_display = df[['Time', 'Hours', 'Temp', 'Oil pressure', 'Vibration Level', 'Fuel', 'RPM', 'Load']].copy()
    df_display.index = df_display.index + 1  # Start index from 1
    st.dataframe(df_display.head(num_rows), use_container_width=True)

# Select a parameter to visualize over time
param = st.selectbox("Select parameter to visualize over hours:", 
                     ['Temp', 'Oil pressure', 'Vibration Level', 'Fuel', 'RPM', 'Load'])

# Plotting
st.line_chart(df.set_index('Hours')[param])

st.markdown("---")
st.subheader("📈 Moving Average Trend Analysis")

# Select parameter to analyze
ma_param = st.selectbox(
    "Select parameter for moving average analysis:",
    ['Temp', 'Oil pressure', 'Vibration Level', 'Fuel', 'RPM', 'Load'],
    key="ma_param"
)

# Select window size
window_size = st.slider("Select moving average window (in hours)", 2, 50, 5)

# Sort by Hours for consistency
df_sorted = df.sort_values("Hours")

# Calculate moving average
df_sorted['Moving Average'] = df_sorted[ma_param].rolling(window=window_size).mean()

# Display chart
st.line_chart(df_sorted.set_index('Hours')[[ma_param, 'Moving Average']])


st.markdown("---")
st.subheader("⚙️ Oil Pressure vs Engine Load")

# Scatter plot using Altair
import altair as alt

scatter_chart = alt.Chart(df).mark_circle(size=60).encode(
    x=alt.X('Load', title='Engine Load (%)'),
    y=alt.Y('Oil pressure', title='Oil Pressure (bar)'),
    color=alt.Color('Temp', scale=alt.Scale(scheme='redyellowblue'), title='Temp (°C)'),
    tooltip=['Hours', 'Load', 'Oil pressure', 'Temp']
).interactive().properties(
    width=700,
    height=400,
    title='Relationship Between Load and Oil Pressure (Colored by Temp)'
)

st.altair_chart(scatter_chart, use_container_width=True)

st.markdown("---")
st.subheader("🧭 Overall Engine Safety Distribution")

# Apply the same safety check to the full dataset
df['Status'] = df.apply(check_safety, axis=1)

# Calculate counts
status_counts = df['Status'].value_counts().reset_index()
status_counts.columns = ['Status', 'Count']

# Pie chart using Plotly
import plotly.express as px

fig_pie = px.pie(
    status_counts,
    names='Status',
    values='Count',
    color='Status',
    color_discrete_map={'Safe': 'green', 'Unsafe': 'red'},
    title='Safe vs Unsafe Engine Records'
)
fig_pie.update_traces(textposition='inside', textinfo='percent+label')

st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("### ℹ️ Project Summary & Notes")
st.markdown("""
This dashboard monitors engine parameters and predicts when maintenance is due using machine learning. It combines:

- **Prediction Model**: Estimates remaining hours before next maintenance using a Random Forest Regressor.
- **Safety Checks**: Flags unsafe engine conditions based on preset threshold ranges (like RPM, oil pressure, temp).
- **Data Visuals**: Shows moving averages, oil pressure trends, and more over time for deeper analysis.

**Key Notes**:
- Model must be trained first and saved as `regression_model.pkl`.
- Index starts from 0 (normal for pandas).
- Use `pip install plotly` to enable charts.
- Avoid using `squared=False` in `mean_squared_error()` — it’s deprecated.
- Ensure your data is clean to avoid poor predictions.

 """)




