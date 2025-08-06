import pandas as pd
import numpy as np
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import hopsworks

# === Step 1: Connect to Hopsworks and Load Data ===
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()

processed_fg = fs.get_feature_group(name="processed_aqi_data_v2", version=1)
df = processed_fg.read()
print("✅ Loaded processed AQI data from Hopsworks")

# === Step 2: Convert to Daily AQI ===
df["timestamp_str"] = pd.to_datetime(df["timestamp_str"])
df["date"] = df["timestamp_str"].dt.date
daily_df = df.groupby("date").agg({"calculated_aqi": "mean"}).reset_index()
daily_df.rename(columns={"calculated_aqi": "aqi"}, inplace=True)

# === Step 3: Create Lag Features and Multi-output Targets ===
for lag in range(1, 8):
    daily_df[f"lag_{lag}"] = daily_df["aqi"].shift(lag)

daily_df["t_plus_1"] = daily_df["aqi"].shift(-1)
daily_df["t_plus_2"] = daily_df["aqi"].shift(-2)
daily_df["t_plus_3"] = daily_df["aqi"].shift(-3)

daily_df.dropna(inplace=True)

features = [f"lag_{i}" for i in range(1, 8)]
X = daily_df[features]
y = daily_df[["t_plus_1", "t_plus_2", "t_plus_3"]]

# === Step 4: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 5: Train Multi-Output Model ===
base_model = RandomForestRegressor(n_estimators=100, random_state=42)
multi_model = MultiOutputRegressor(base_model)
multi_model.fit(X_train, y_train)

# === Step 6: Evaluate ===
y_pred = multi_model.predict(X_test)

for i, label in enumerate(["Day +1", "Day +2", "Day +3"]):
    print(f"\n📊 {label} AQI Prediction")
    print(f"📉 MAE: {mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]):.2f}")
    print(f"📈 R² Score: {r2_score(y_test.iloc[:, i], y_pred[:, i]):.2f}")

# === Step 7: Forecast Next 3 Days ===
latest_input = daily_df[features].iloc[-1:]
forecast = multi_model.predict(latest_input)[0]

print("\n🔮 Forecast for Next 3 Days AQI:")
print(f"Day +1 AQI: {forecast[0]:.2f}")
print(f"Day +2 AQI: {forecast[1]:.2f}")
print(f"Day +3 AQI: {forecast[2]:.2f}")

# === Step 8: Optional - Plot Actual vs Predicted for Day +1 ===
plt.figure(figsize=(8, 5))
plt.plot(y_test.iloc[:30, 0].values, label="Actual Day+1")
plt.plot(y_pred[:30, 0], label="Predicted Day+1")
plt.title("Multi-Output RF - AQI Prediction (Day +1)")
plt.xlabel("Sample")
plt.ylabel("AQI")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
