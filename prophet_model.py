import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import hopsworks
from sklearn.metrics import r2_score

# === Step 1: Connect to Hopsworks and Load Processed Data ===
project = hopsworks.login(api_key_value="COunaWRa4xAVurwN.ilcGS68HagjlXHSzceROm78ktOrAt4BrSqRnP2GwvMRKor9nPV9TTGuMAG3TSmh7")
fs = project.get_feature_store()
processed_fg = fs.get_feature_group(name="processed_aqi_data_v2", version=1)
df = processed_fg.read()
print("✅ Loaded processed AQI data from Hopsworks")

# === Step 2: Prepare Data ===
df["timestamp_str"] = pd.to_datetime(df["timestamp_str"]).dt.tz_localize(None)
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

model.fit(df_prophet)

# === Step 5: Predict on Training Data for R² ===
train_forecast = model.predict(df_prophet)
r2_train = r2_score(df_prophet["y"], train_forecast["yhat"])
print(f"\n📊 Training R² Score: {r2_train:.4f}")

# === Step 6: Forecast Next 3 Days ===
future = model.make_future_dataframe(periods=3)
last_row = df_prophet.iloc[-1]
for reg in regressors:
    future[reg] = [last_row[reg]] * len(future)

forecast = model.predict(future)

# === Step 7: Calculate R² for the next 3 days if actuals exist ===
future_actuals = df_prophet[df_prophet["ds"].isin(forecast["ds"].tail(3))]
if not future_actuals.empty:
    # Align predictions with actuals
    merged = forecast[["ds", "yhat"]].merge(future_actuals[["ds", "y"]], on="ds", how="inner")
    r2_next3 = r2_score(merged["y"], merged["yhat"])
    print(f"📊 R² Score for Next 3 Days: {r2_next3:.4f}")
else:
    print("⚠️ No actual AQI data available for the next 3 days to compute R².")

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
