import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import hopsworks

# === Step 1: Connect to Hopsworks and Load Data ===
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()

processed_fg = fs.get_feature_group(name="processed_aqi_data_v2", version=1)
df = processed_fg.read()
print("✅ Loaded processed AQI data from Hopsworks")

# === Step 2: Use All Scaled Features Except timestamp_str ===
df["date"] = pd.to_datetime(df["timestamp_str"])
df = df.sort_values("date").reset_index(drop=True)

# Select only scaled features for training
features = [col for col in df.columns if "_scaled" in col]
X = df[features]
y = df["calculated_aqi"]

# === Step 3: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Step 4: Train XGBoost Regressor ===
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Step 5: Evaluate ===
y_pred = model.predict(X_test)

print("\n📊 Evaluation on Test Set")
print(f"📉 MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"📈 R² Score: {r2_score(y_test, y_pred):.2f}")

# === Step 6: Recursive Forecast for Next 3 Days ===
next_preds = []
latest_input = X.iloc[-1].copy()

for day in range(3):
    pred = model.predict(latest_input.values.reshape(1, -1))[0]
    next_preds.append(pred)

    # (Optional) If AQI affects features, you can simulate feature changes here

print("\n📅 Recursive Forecast for Next 3 Days AQI:")
for i, p in enumerate(next_preds, 1):
    print(f"🔮 Day +{i} AQI: {p:.2f}")

# === Step 7: Plot ===
plt.figure(figsize=(8, 5))
plt.plot(y_test.values[:30], label="Actual")
plt.plot(y_pred[:30], label="Predicted")
plt.title("XGBoost - AQI Prediction")
plt.xlabel("Sample")
plt.ylabel("AQI")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
