import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.multioutput import MultiOutputRegressor
import hopsworks

# === Connect and load data ===
project = hopsworks.login(api_key_value="COunaWRa4xAVurwN.ilcGS68HagjlXHSzceROm78ktOrAt4BrSqRnP2GwvMRKor9nPV9TTGuMAG3TSmh7")
fs = project.get_feature_store()
fg = fs.get_feature_group(name="processed_aqi_data_v2", version=1)
df = fg.read()
df['timestamp_str'] = pd.to_datetime(df['timestamp_str'])
df = df.sort_values("timestamp_str").reset_index(drop=True)

# Feature engineering (lags, rolling stats, cyclical encoding, targets)
df['AQI_t-1'] = df['calculated_aqi'].shift(1)
df['AQI_t-2'] = df['calculated_aqi'].shift(2)
df['AQI_t-3'] = df['calculated_aqi'].shift(3)

df['AQI_roll7_mean'] = df['calculated_aqi'].rolling(window=7).mean().shift(1)
df['AQI_roll14_mean'] = df['calculated_aqi'].rolling(window=14).mean().shift(1)
df['AQI_roll7_std'] = df['calculated_aqi'].rolling(window=7).std().shift(1)

df['hour_sin'] = np.sin(2 * np.pi * df['hour_scaled'])
df['hour_cos'] = np.cos(2 * np.pi * df['hour_scaled'])
df['day_sin'] = np.sin(2 * np.pi * df['day_scaled'] / 31)
df['day_cos'] = np.cos(2 * np.pi * df['day_scaled'] / 31)
df['month_sin'] = np.sin(2 * np.pi * df['month_scaled'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month_scaled'] / 12)

df['AQI+1'] = df['calculated_aqi'].shift(-1)
df['AQI+2'] = df['calculated_aqi'].shift(-2)
df['AQI+3'] = df['calculated_aqi'].shift(-3)

df = df.dropna().reset_index(drop=True)

features = [
    'AQI_t-1', 'AQI_t-2', 'AQI_t-3',
    'AQI_roll7_mean', 'AQI_roll14_mean', 'AQI_roll7_std',
    'co_scaled', 'no_log_scaled', 'no2_scaled', 'o3_scaled',
    'so2_log_scaled', 'nh3_log_scaled', 'aqi_change_rate_scaled',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
]

# For tuning, use AQI+1 target only
X = df[features]
y = df['AQI+1']

# Train-test split (80-20)
split_ratio = 0.8
split_idx = int(len(X) * split_ratio)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# XGBoost model and hyperparameter space
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

param_dist = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'n_estimators': [100, 200, 300, 500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3, 0.5]
}

# Time series cross-validator
tscv = TimeSeriesSplit(n_splits=5)
r2_scorer = make_scorer(r2_score)

from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring=r2_scorer,
    cv=tscv,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Run hyperparameter tuning
random_search.fit(X_train, y_train)

print("\nBest hyperparameters found:")
print(random_search.best_params_)
print(f"Best CV R² score: {random_search.best_score_:.4f}")

# Retrain multi-output model with best params
best_params = random_search.best_params_
base_model = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
multi_model = MultiOutputRegressor(base_model)

# Targets for all 3 horizons
y_multi = df[['AQI+1', 'AQI+2', 'AQI+3']]
multi_model.fit(X_train, y_multi.iloc[:split_idx])

# Evaluate on training set
y_train_multi = y_multi.iloc[:split_idx]
y_pred_train = multi_model.predict(X_train)
print("\n=== Training R² Scores ===")
for i, col in enumerate(['AQI+1', 'AQI+2', 'AQI+3']):
    r2_train = r2_score(y_train_multi[col], y_pred_train[:, i])
    print(f"Train R² for {col}: {r2_train:.4f}")

# Evaluate on test set
y_test_multi = y_multi.iloc[split_idx:]
y_pred_test = multi_model.predict(X_test)
print("\n=== Testing R² Scores ===")
for i, col in enumerate(['AQI+1', 'AQI+2', 'AQI+3']):
    r2_test = r2_score(y_test_multi[col], y_pred_test[:, i])
    print(f"Test R² for {col}: {r2_test:.4f}")
