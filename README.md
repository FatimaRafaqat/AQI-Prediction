# 🌬️ Air Quality Index (AQI) Prediction System

“Clean air meets clean code.”
A fully serverless machine learning pipeline that predicts the Air Quality Index (AQI) for the next 3 days using real-time environmental data from the OpenWeather API.

# 📖 Overview

Air pollution is a growing concern that affects human health and the environment.
This project aims to predict the Air Quality Index (AQI) for upcoming days using historical and real-time pollutant data.

It automates the entire process — from data collection to dashboard visualization — using a serverless CI/CD pipeline and machine learning models deployed with FastAPI.

# 🚀 Features

✅ Automated Data Collection

Fetches air quality data hourly via the OpenWeather API using GitHub Actions.

✅ Data Management with Hopsworks Feature Store

Stores, versions, and manages all raw and processed features for reproducibility.

✅ Data Preprocessing & Feature Engineering

Handles outliers, normalization, and transformations.

Adds temporal features (hour, day, month) and AQI rate of change.

✅ Model Training & Forecasting

Trains Linear Regression, Random Forest, and XGBoost models.

Implements a Multi-Output Regressor to predict AQI for the next 3 days.

✅ Deployment & Visualization

Deploys prediction results through FastAPI endpoints.

Displays AQI trends on a real-time dashboard.

✅ Continuous Integration & Delivery (CI/CD)

Fully automated workflow — hourly data updates and daily retraining with zero manual intervention.

# 📊 Model Performance
| Model                  | R² Score  | Notes                                   |
| ---------------------- | --------- | --------------------------------------- |
| Linear Regression      | 0.85–0.90 | Baseline model                          |
| Random Forest          | > 0.90    | Handles non-linearity effectively       |
| XGBoost (Multi-Output) | > 0.95    | Best performance across 3-day forecasts |


✅ Final Model: Multi-Output XGBoost
✅ Forecast Range: Next 3 Days
✅ Accuracy: R² above 95%

# ⚙️ Workflow Architecture

# Data Collection (Hourly)

OpenWeather API → GitHub Actions → Hopsworks Feature Store

# Preprocessing & Feature Engineering (Daily)

Outlier treatment, scaling, temporal features, and AQI computation

# Model Training & Prediction (Daily)

Trains multiple models and selects the best-performing one

# API & Dashboard Deployment

FastAPI fetches predictions from Hopsworks and serves them to the dashboard


# 🤝 Acknowledgments

This project was developed as part of my internship at 10 pearl Institute
.
A heartfelt thanks to my mentorsAbdullah Farooqi, Muhammad Faizan Owais, and Ahmed Mozammil Iqbal for their guidance and support throughout this journey.


# 🧩 Future Improvements

Add deep learning models (LSTM / GRU) for sequence prediction.

Include multiple cities for global AQI comparison.

Build an interactive dashboard with user-selected timeframes.

# 🏅 Author

👩‍💻 Fatima Rafaqat
Data Science & Machine Learning Enthusiast

📧 fatimarafaqat2000@gmail.com

🌐 https://www.linkedin.com/in/fatimarafaqat/



