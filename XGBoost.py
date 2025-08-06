import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import hopsworks

# === Step 1: Connect to Hopsworks and Load Processed Data ===
project = hopsworks.login(api_key_value="COunaWRa4xAVurwN.ilcGS68HagjlXHSzceROm78ktOrAt4BrSqRnP2GwvMRKor9nPV9TTGuMAG3TSmh7")
fs = project.get_feature_store()

processed_fg = fs.get_feature_group(name="processed_aqi_data_v2", version=1)
df = processed_fg.read()
print("✅ Loaded processed AQI data from Hopsworks")

# === Step 2: Preprocess ===
if "timestamp_str" in df.columns:
    df = df.drop(columns=["timestamp_str"])

# Define features and target
target_col = "calculated_aqi"
feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols]
y = df[target_col]

# === Step 3: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 4: Train XGBoost Regressor ===
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

# === Step 5: Evaluate Model ===
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== XGBoost Evaluation ==========")
print(f"📉 MAE: {mae:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# === Step 6: Plot Actual vs Predicted ===
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI (XGBoost)")
plt.grid(True)
plt.tight_layout()
plt.show()
