import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import hopsworks
import os

# === Step 1: Connect to Hopsworks and Load Data ===
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()

# Load processed feature group
processed_fg = fs.get_feature_group(name="processed_aqi_data_v2", version=1)
df = processed_fg.read()
print("✅ Loaded processed AQI data from Hopsworks")

# === Step 2: Prepare Features and Target ===
feature_cols = [col for col in df.columns if "_scaled" in col]
X = df[feature_cols]
y = df["calculated_aqi"]

# === Step 3: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 4: Train Linear Regression Model ===
lr = LinearRegression()
lr.fit(X_train, y_train)

# === Step 5: Make Predictions ===
y_pred = lr.predict(X_test)

# === Step 6: Evaluate Model ===
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== Linear Regression Evaluation ==========")
print(f"📉 Mean Squared Error (MSE): {mse:.2f}")
print(f"📉 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# === Step 7: Plot Actual vs Predicted AQI ===
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI (Linear Regression)")
plt.grid(True)
plt.tight_layout()
plt.show()
