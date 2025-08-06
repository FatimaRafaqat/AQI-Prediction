import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet  # ✅ Correct import
import hopsworks

# === Step 1: Connect to Hopsworks and Load Processed Data ===
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()
processed_fg = fs.get_feature_group(name="processed_aqi_data_v2", version=1)
df = processed_fg.read()
print("✅ Loaded processed AQI data from Hopsworks")

# === Step 2: Prepare Data ===
df["timestamp_str"] = pd.to_datetime(df["timestamp_str"]).dt.tz_localize(None)  # ✅ Remove timezone
df = df.sort_values("timestamp_str")

# Rename for Prophet
df_prophet = df.rename(columns={"timestamp_str": "ds", "calculated_aqi": "y"})

# === Step 3: Define Regressor Columns ===
regressors = [
    "co_scaled", "no_log_scaled", "no2_scaled", "o3_scaled",
    "so2_log_scaled", "nh3_log_scaled", "aqi_change_rate_scaled",
    "hour_scaled", "day_scaled", "month_scaled"
]

# Keep only required columns
df_prophet = df_prophet[["ds", "y"] + regressors]

# === Step 4: Initialize Prophet and Add Regressors ===
model = Prophet(daily_seasonality=True)
for reg in regressors:
    model.add_regressor(reg)

# === Step 5: Fit Model ===
model.fit(df_prophet)

# === Step 6: Forecast Next 3 Days ===
future = model.make_future_dataframe(periods=3)

# Copy last known regressor values forward
last_row = df_prophet.iloc[-1]
for reg in regressors:
    future[reg] = [last_row[reg]] * len(future)

# === Step 7: Predict ===
forecast = model.predict(future)

# === Step 8: Plot Forecast ===
model.plot(forecast)
plt.title("📈 AQI Forecast with Prophet")
plt.xlabel("Date")
plt.ylabel("Predicted AQI")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 9: Print Forecast Results ===
print("\n📊 Next 3-Day AQI Forecast:")
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(3))
