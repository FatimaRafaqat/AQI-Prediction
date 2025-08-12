import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import hopsworks

# === Step 1: Connect to Hopsworks and Load Processed Data ===
project = hopsworks.login(api_key_value="COunaWRa4xAVurwN.ilcGS68HagjlXHSzceROm78ktOrAt4BrSqRnP2GwvMRKor9nPV9TTGuMAG3TSmh7")
fs = project.get_feature_store()
processed_fg = fs.get_feature_group(name="processed_aqi_data_v2", version=1)
df = processed_fg.read()
print("✅ Loaded processed AQI data from Hopsworks")

# === Step 2: Preprocessing - Create lag features and targets ===
df['timestamp_str'] = pd.to_datetime(df['timestamp_str'])
df = df.sort_values('timestamp_str').reset_index(drop=True)

# Lag features (AQI of previous 1,2,3 days)
df['AQI_t-1'] = df['calculated_aqi'].shift(1)
df['AQI_t-2'] = df['calculated_aqi'].shift(2)
df['AQI_t-3'] = df['calculated_aqi'].shift(3)

# Targets: AQI for next 3 days
df['AQI+1'] = df['calculated_aqi'].shift(-1)
df['AQI+2'] = df['calculated_aqi'].shift(-2)
df['AQI+3'] = df['calculated_aqi'].shift(-3)

# Drop rows with NaNs created by shifting
df = df.dropna().reset_index(drop=True)

# Define features and targets
feature_cols = ['AQI_t-1', 'AQI_t-2', 'AQI_t-3'] + [col for col in df.columns if col not in ['calculated_aqi', 'AQI+1', 'AQI+2', 'AQI+3', 'timestamp_str', 'AQI_t-1', 'AQI_t-2', 'AQI_t-3']]
target_cols = ['AQI+1', 'AQI+2', 'AQI+3']

X = df[feature_cols]
y = df[target_cols]

# === Step 3: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Step 4: Train MultiOutput XGBoost Regressor ===
base_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='reg:squarederror',
    random_state=42
)

model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

# === Step 5: Evaluate model ===
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("=== Training R² Scores ===")
for i, target in enumerate(target_cols):
    r2_train = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
    print(f"{target} — Training R²: {r2_train:.3f}")

print("\n=== Testing R² Scores ===")
for i, target in enumerate(target_cols):
    r2_test = r2_score(y_test.iloc[:, i], y_test_pred[:, i])
    print(f"{target} — Testing R²: {r2_test:.3f}")

# Also print MAE on test set for reference
print("\n=== Testing MAE Scores ===")
for i, target in enumerate(target_cols):
    mae_test = mean_absolute_error(y_test.iloc[:, i], y_test_pred[:, i])
    print(f"{target} — Testing MAE: {mae_test:.2f}")

# === Step 6: Predict next 3 days AQI from the latest data ===
last_features = X.iloc[[-1]]
future_preds = model.predict(last_features)
future_preds = np.round(future_preds[0], 2)  # 1D array of [day1, day2, day3]
print("\nNext 3 days AQI predictions:", future_preds)

# === Step 7: Store predictions into Hopsworks ===
from datetime import datetime, timedelta

# Build DataFrame for insertion
today = pd.Timestamp.utcnow().normalize()
dates = [today + pd.Timedelta(days=i+1) for i in range(3)]  # next 3 days

pred_df = pd.DataFrame({
    "prediction_date": [d.strftime("%Y-%m-%d") for d in dates],  # store as string
    "prediction_value": future_preds,
    "horizon_day": [1, 2, 3],
    "model_version": ["v1.0"] * 3,
    "prediction_ts": pd.Timestamp.utcnow()  # event time can still be timestamp
})

# Create (or get) the Feature Group
pred_fg = fs.get_or_create_feature_group(
    name="model_predictions",
    version=1,
    primary_key=["prediction_date"],  # now string type
    description="3-day AQI forecasts",
    online_enabled=True,
    event_time="prediction_ts"
)

# Insert into Feature Group
pred_fg.insert(pred_df, write_options={"wait_for_job": False})
print("✅ Predictions uploaded to Hopsworks Feature Group 'model_predictions'")
