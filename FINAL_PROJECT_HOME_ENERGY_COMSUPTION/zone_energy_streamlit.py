# --- energy_prediction.py code (now Streamlit batch evaluation section) ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

with st.expander("Batch Model Evaluation on home_energy.csv", expanded=False):
    if os.path.exists('home_energy.csv'):
        df = pd.read_csv('home_energy.csv', parse_dates=['time'])
        df = df.sort_values('time')
        df = df.ffill()
        df['hour'] = df['time'].dt.hour
        df['dayofweek'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        feature_cols = ['hour', 'dayofweek', 'month', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        target_col = 'Global_active_power'
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        st.markdown(f"""
        **Model Evaluation Report**
        - Mean Absolute Error (MAE): `{mae:.4f}`
        - Mean Squared Error (MSE): `{mse:.4f}`
        - Root Mean Squared Error (RMSE): `{rmse:.4f}`
        - R² Score: `{r2:.4f}`
        """)
        # Actual vs Predicted chart
        fig1, ax1 = plt.subplots(figsize=(8,3))
        ax1.plot(y_test.values, label='Actual')
        ax1.plot(y_pred, label='Predicted')
        ax1.legend()
        ax1.set_title('Actual vs Predicted Hourly Energy Consumption')
        ax1.set_xlabel('Time (test set)')
        ax1.set_ylabel('Global Active Power')
        st.pyplot(fig1)
        # Residuals plot
        residuals = y_test.values - y_pred
        fig2, ax2 = plt.subplots(figsize=(8,2))
        ax2.scatter(range(len(residuals)), residuals, alpha=0.5)
        ax2.axhline(0, color='red', linestyle='--')
        ax2.set_title('Residuals (Actual - Predicted)')
        ax2.set_xlabel('Test Sample Index')
        ax2.set_ylabel('Residual')
        st.pyplot(fig2)
        # Example prediction
        example = pd.DataFrame({
            'hour': [14],
            'dayofweek': [2],
            'month': [6],
            'Voltage': [235],
            'Global_intensity': [15],
            'Sub_metering_1': [0],
            'Sub_metering_2': [1],
            'Sub_metering_3': [17]
        })
        future_pred = model.predict(example)
        st.info(f"Predicted global active power for example input: {future_pred[0]:.2f}")
    else:
        st.warning("home_energy.csv not found in the project directory. Batch evaluation is skipped.")

# --- End of energy_prediction.py code ---

# --- zone_energy_streamlit.py code (Streamlit app) ---
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import os

# Set Streamlit page config for fixed width and better alignment
st.set_page_config(page_title="AI Smart Energy Consumption Prediction", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main {
        max-width: 700px;
        margin: auto;
        padding-top: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stSelectbox, .stMultiSelect, .stNumberInput {
        width: 100% !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("AI Smart Energy Consumption Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload Tetuan City Power Consumption CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
elif os.path.exists('Tetuan_City_power_consumption.csv'):
    df = pd.read_csv('Tetuan_City_power_consumption.csv')
    st.info("Using Tetuan_City_power_consumption.csv from project folder.")
else:
    st.warning("Please upload the Tetuan City Power Consumption CSV file to proceed.")
    st.stop()

# Show preview and columns
with st.expander("Preview of uploaded data", expanded=False):
    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"Columns detected: {list(df.columns)}")

# Only allow numeric columns as features and targets
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_columns:
    st.error("No numeric columns found in the uploaded file. Please upload a valid dataset.")
    st.stop()

with st.form("feature_target_form"):
    st.subheader("Model Configuration")
    zones = st.multiselect("Select zone/target columns", options=numeric_columns, default=numeric_columns[-3:])
    features = st.multiselect("Select feature columns (weather, etc.)", options=[col for col in numeric_columns if col not in zones], default=[col for col in numeric_columns if col not in zones][:5])
    submitted = st.form_submit_button("Train Model")

if not features or not zones:
    st.warning("Please select at least one feature and one target column.")
    st.stop()

# Train/test split and model training for each zone
models = {}
r2_scores_train = {}
r2_scores_test = {}
for zone in zones:
    X = df[features]
    y = df[zone]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    models[zone] = model
    r2_scores_train[zone] = r2_train
    r2_scores_test[zone] = r2_test

st.subheader("Prediction Interface")
zone = st.selectbox("Select Zone for Prediction", zones, key="zone_select")

if zone in r2_scores_test:
    st.info(f"R² (Train) for {zone}: {r2_scores_train[zone]:.4f}")
    st.info(f"R² (Test) for {zone}: {r2_scores_test[zone]:.4f}")
    if r2_scores_test[zone] < 0.9:
        st.warning(f"Warning: R² (Test) for {zone} is below 0.9. Try selecting different features or check your data quality.")

with st.form("prediction_form"):
    st.markdown("**Enter feature values for prediction:**")
    cols = st.columns(len(features))
    input_data = {}
    for i, feat in enumerate(features):
        with cols[i]:
            input_data[feat] = st.number_input(feat, value=float(df[feat].mean()) if pd.api.types.is_numeric_dtype(df[feat]) else 0.0, key=f"input_{feat}")
    predict_btn = st.form_submit_button("Predict Energy Consumption")

if 'input_data' in locals() and predict_btn:
    input_df = pd.DataFrame([input_data])
    pred = models[zone].predict(input_df)[0]
    st.success(f"Predicted {zone}: {pred:.2f} kW")

with st.expander("Show historical data for selected zone", expanded=False):
    st.line_chart(df[[zone]]) 