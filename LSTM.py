import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, r2_score
import hopsworks

# === Step 1: Connect to Hopsworks and Load Data ===
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()

processed_fg = fs.get_feature_group(name="processed_aqi_data_v2", version=1)
df = processed_fg.read()
print("✅ Loaded processed AQI data from Hopsworks")

# === Step 2: Preprocess ===
if "timestamp_str" in df.columns:
    df = df.drop(columns=["timestamp_str"])

# Sort by index or timestamp if needed
df = df.sort_index()

# Normalize target if needed (optional)
target_col = "calculated_aqi"
feature_cols = [col for col in df.columns if col != target_col]

# === Step 3: Create Sequences ===
def create_sequences(data, target, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(target[i+seq_len])
    return np.array(X), np.array(y)

sequence_length = 7
X_seq, y_seq = create_sequences(df[feature_cols].values, df[target_col].values, sequence_length)

# Train-test split
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# === Step 4: Build LSTM Model ===
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# === Step 5: Train Model ===
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1, verbose=1)

# === Step 6: Evaluate ===
y_pred = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== LSTM Evaluation ==========")
print(f"📉 MAE: {mae:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# === Step 7: Plot ===
plt.figure(figsize=(10, 5))
plt.plot(y_test[:50], label="Actual")
plt.plot(y_pred[:50], label="Predicted")
plt.title("LSTM AQI Prediction (Next Day)")
plt.xlabel("Sample")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
