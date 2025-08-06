import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import hopsworks

# === Step 1: Connect to Hopsworks and Load Data ===
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()

processed_fg = fs.get_feature_group(name="processed_aqi_data_v2", version=1)
df = processed_fg.read()
print("✅ Loaded processed AQI data from Hopsworks")

# === Step 2: Convert to Daily AQI and Create Lags ===
df["date"] = pd.to_datetime(df["timestamp_str"])  # FIXED line
daily_df = df.groupby("date").agg({"calculated_aqi": "mean"}).reset_index()
daily_df.rename(columns={"calculated_aqi": "aqi"}, inplace=True)


# Create lag features (last 7 days AQI)
for lag in range(1, 8):
    daily_df[f"lag_{lag}"] = daily_df["aqi"].shift(lag)

# Targets: predict next 3 days AQI
daily_df["t_plus_1"] = daily_df["aqi"].shift(-1)
daily_df["t_plus_2"] = daily_df["aqi"].shift(-2)
daily_df["t_plus_3"] = daily_df["aqi"].shift(-3)

# Drop rows with NaN (due to shift)
daily_df = daily_df.dropna().reset_index(drop=True)

# === Step 3: Train-Test Split ===
features = [f"lag_{i}" for i in range(1, 8)]
X = daily_df[features]
y1 = daily_df["t_plus_1"]
y2 = daily_df["t_plus_2"]
y3 = daily_df["t_plus_3"]

X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=42)
_, _, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=42)
_, _, y3_train, y3_test = train_test_split(X, y3, test_size=0.2, random_state=42)

# === Step 4: Train Random Forest Regressors ===
rf1 = RandomForestRegressor(n_estimators=100, random_state=42)
rf2 = RandomForestRegressor(n_estimators=100, random_state=42)
rf3 = RandomForestRegressor(n_estimators=100, random_state=42)

rf1.fit(X_train, y1_train)
rf2.fit(X_train, y2_train)
rf3.fit(X_train, y3_train)

# === Step 5: Evaluate ===
def evaluate(y_true, y_pred, label):
    print(f"\n📊 {label} AQI Prediction")
    print(f"📉 MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"📈 R² Score: {r2_score(y_true, y_pred):.2f}")

evaluate(y1_test, rf1.predict(X_test), "Day +1")
evaluate(y2_test, rf2.predict(X_test), "Day +2")
evaluate(y3_test, rf3.predict(X_test), "Day +3")

# === Step 6: Predict the Next 3 Days AQI ===
latest_lags = daily_df[features].iloc[-1:].values
next_1 = rf1.predict(latest_lags)[0]
next_2 = rf2.predict(latest_lags)[0]
next_3 = rf3.predict(latest_lags)[0]

print("\n📅 Forecast for Next 3 Days AQI:")
print(f"🔮 Day +1 AQI: {next_1:.2f}")
print(f"🔮 Day +2 AQI: {next_2:.2f}")
print(f"🔮 Day +3 AQI: {next_3:.2f}")

# === Step 7: Plot Actual vs Predicted on Test Set ===
plt.figure(figsize=(8, 5))
plt.plot(y1_test.values[:30], label="Actual Day+1")
plt.plot(rf1.predict(X_test)[:30], label="Predicted Day+1")
plt.title("Random Forest - AQI Prediction (Next Day)")
plt.xlabel("Sample")
plt.ylabel("AQI")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
