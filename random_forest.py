import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import hopsworks
import os

# === Step 1: Connect to Hopsworks and Load Processed Data ===
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()

processed_fg = fs.get_feature_group(name="processed_aqi_data_v2", version=1)
df = processed_fg.read()
print("✅ Loaded processed AQI data from Hopsworks")

# === Step 2: Aggregate by day ===
df['timestamp_str'] = pd.to_datetime(df['timestamp_str'])
df['date'] = df['timestamp_str'].dt.date
daily_df = df.groupby('date').agg({'calculated_aqi': 'mean'}).reset_index()
daily_df.rename(columns={"calculated_aqi": "aqi"}, inplace=True)

# === Step 3: Create Lag Features and Forecast Targets ===
for lag in range(1, 8):
    daily_df[f"lag_{lag}"] = daily_df["aqi"].shift(lag)

# Targets: predict next 3 days AQI
daily_df["t_plus_1"] = daily_df["aqi"].shift(-1)
daily_df["t_plus_2"] = daily_df["aqi"].shift(-2)
daily_df["t_plus_3"] = daily_df["aqi"].shift(-3)

daily_df = daily_df.dropna().reset_index(drop=True)

# === Step 4: Train-Test Split ===
features = [f"lag_{i}" for i in range(1, 8)]
X = daily_df[features]
y1 = daily_df["t_plus_1"]
y2 = daily_df["t_plus_2"]
y3 = daily_df["t_plus_3"]

X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=42)
_, _, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=42)
_, _, y3_train, y3_test = train_test_split(X, y3, test_size=0.2, random_state=42)

# === Step 5: Train Random Forest Models ===
rf1 = RandomForestRegressor(n_estimators=100, random_state=42)
rf2 = RandomForestRegressor(n_estimators=100, random_state=42)
rf3 = RandomForestRegressor(n_estimators=100, random_state=42)

rf1.fit(X_train, y1_train)
rf2.fit(X_train, y2_train)
rf3.fit(X_train, y3_train)

# === Step 6: Make Predictions ===
y1_pred = rf1.predict(X_test)
y2_pred = rf2.predict(X_test)
y3_pred = rf3.predict(X_test)

# === Step 7: Evaluate ===
def evaluate(y_true, y_pred, label):
    print(f"\n📊 {label} AQI Prediction")
    print(f"📉 MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"📈 R² Score: {r2_score(y_true, y_pred):.2f}")

evaluate(y1_test, y1_pred, "Day +1")
evaluate(y2_test, y2_pred, "Day +2")
evaluate(y3_test, y3_pred, "Day +3")

# === Optional: Plot Prediction ===
plt.figure(figsize=(8, 5))
plt.plot(y1_test.values[:30], label="Actual Day+1")
plt.plot(y1_pred[:30], label="Predicted Day+1")
plt.title("Random Forest - AQI Prediction (Next Day)")
plt.xlabel("Sample")
plt.ylabel("AQI")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
