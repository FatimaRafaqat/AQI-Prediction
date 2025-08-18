import pandas as pd
from datetime import datetime
import hopsworks

# Connect to Hopsworks project
project = hopsworks.login(api_key_value="YOUR_API_KEY_HERE")
fs = project.get_feature_store()

# Create or get the feature group for model predictions
pred_fg = fs.get_or_create_feature_group(
    name="aqi_model_predictions",
    version=1,
    description="Stores next 3 days AQI predictions and evaluation metrics for multiple models",
    primary_key=["timestamp", "model_name"],  # Composite key to store multiple models per timestamp
    event_time="timestamp"
)

# Prepare a sample prediction results record (you will generate these in your workflow)
record = {
    "timestamp": datetime.now(),
    "model_name": "XGBoost",
    "pred_day_1": 150.5,
    "pred_day_2": 140.3,
    "pred_day_3": 130.8,
    "r2_score": 0.78,
    "mae": 12.4,
}

# Convert to DataFrame
df_record = pd.DataFrame([record])

# Insert into Feature Group
pred_fg.insert(df_record, write_options={"wait_for_job": False})

print("✅ Prediction results inserted into Hopsworks feature group")
