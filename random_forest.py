import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import hopsworks
import os

# === Step 1: Connect to Hopsworks and Load Processed Data ===
project = hopsworks.login(api_key_value="COunaWRa4xAVurwN.ilcGS68HagjlXHSzceROm78ktOrAt4BrSqRnP2GwvMRKor9nPV9TTGuMAG3TSmh7")
fs = project.get_feature_store()

processed_fg = fs.get_feature_group(name="processed_aqi_data_v2", version=1)
df = processed_fg.read()
print("✅ Loaded processed AQI data from Hopsworks")

# === Step 2: Drop timestamp_str ===
if "timestamp_str" in df.columns:
    df = df.drop(columns=["timestamp_str"])

# === Step 3: Prepare Features and Target ===
feature_cols = [col for col in df.columns if col != "calculated_aqi"]
X = df[feature_cols]
y = df["calculated_aqi"]

# === Step 4: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 5: Train Random Forest Model ===
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# === Step 6: Make Predictions ===
y_pred = rf.predict(X_test)

# === Step 7: Evaluate Model ===
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== Random Forest Evaluation ==========")
print(f"📉 Mean Squared Error (MSE): {mse:.2f}")
print(f"📉 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# === Step 8: Plot Actual vs Predicted AQI ===
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI (Random Forest)")
plt.grid(True)
plt.tight_layout()
plt.show()
