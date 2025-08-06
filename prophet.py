import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import hopsworks

# === Step 1: Connect to Hopsworks and Load Processed Data ===
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()
processed_fg = fs.get_feature_group(name="processed_aqi_data_v2", version=1)
df = processed_fg.read()
print("✅ Loaded processed AQI data from Hopsworks")

# === Step 2: Prepare Data ===
df["timestamp_str"] = pd.to_datetime(df["timestamp_str"])
df = df.sort_values("timestamp_str")

# Rename for Prophet
df_prophet = df.rename(columns={"timestamp_str": "ds", "calculated_aqi": "y"})

# === Step 3: Define feature columns ===
regressors = [
    "co_scaled", "no_log_scaled", "no2_scaled", "o3_scaled",
    "so2_log_scaled", "nh3_log_scaled", "aqi_change_rate_scaled",
    "hour_scaled", "day_scaled", "month_scaled"
]

# Keep only necessary columns
df_prophet = df_prophet[["ds", "y"] + regressors]

# === Step 4: Initialize and Add Regressors ===
model = Prophet(daily_seasonality=True)
for reg in regressors:
    model.add_regressor(reg)

# === Step 5: Fit Model ===
model.fit(df_prophet)

# === Step 6: Forecast for Next 3 Days ===
future = model.make_future_dataframe(periods=3)

# Append last known values for all regressors
last_row = df_prophet.iloc[-1]
for reg in regressors:
    future[reg] = [last_row[reg]] * len(future)

# === Step 7: Predict and Plot ===
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title("📈 AQI Forecast (with All Features)")
plt.xlabel("Date")
plt.ylabel("Predicted AQI")
plt.tight_layout()
plt.grid(True)
plt.show()

# Show 3-day forecast
print("\n📊 Next 3-Day Forecast:")
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(3))
