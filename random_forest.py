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

# === Step 2: Convert to Daily AQI and Create Lag Features ===
df["date"] = pd.to_datetime(df["timestamp_str"])
daily_df = df.groupby("date").agg({"calculated_aqi": "mean"}).reset_index()
daily_df.rename(columns={"calculated_aqi": "aqi"}, inplace=True)

# Create 7 lag features
for lag in range(1, 8):
    daily_df[f"lag_{lag}"] = daily_df["aqi"].shift(lag)

# Drop NaN rows due to shifting
daily_df = daily_df.dropna().reset_index(drop=True)

# === Step 3: Train Random Forest Model ===
features = [f"lag_{i}" for i in range(1, 8)]
X = daily_df[features]
y = daily_df["aqi"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# === Step 4: Evaluate on Test Set ===
y_pred = rf.predict(X_test)

print("\n📊 Evaluation on Test Set")
print(f"📉 MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"📈 R² Score: {r2_score(y_test, y_pred):.2f}")

# === Step 5: Recursive Prediction for Next 3 Days ===
latest_row = daily_df[features].iloc[-1].values.tolist()
next_3_preds = []

for day in range(3):
    pred = rf.predict([latest_row[-7:]])[0]
    next_3_preds.append(pred)
    latest_row.append(pred)  # shift prediction into lags

# === Step 6: Show Predictions ===
print("\n📅 Recursive Forecast for Next 3 Days AQI:")
for i, p in enumerate(next_3_preds, 1):
    print(f"🔮 Day +{i} AQI: {p:.2f}")

# === Step 7: Optional Plot ===
plt.figure(figsize=(8, 5))
plt.plot(y_test.values[:30], label="Actual")
plt.plot(y_pred[:30], label="Predicted")
plt.title("Random Forest - AQI Prediction")
plt.xlabel("Sample")
plt.ylabel("AQI")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
