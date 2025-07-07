import requests
import pandas as pd
from datetime import datetime

# Coordinates for Islamabad
LAT = 33.6844
LON = 73.0479

# 🔑 Paste your OpenWeather API key here
API_KEY = "509416d5d847f26c0c22b3885774710d"  # Replace with your actual key

def fetch_air_pollution(lat, lon, api_key):
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)

    if response.status_code != 200:
        print("❌ Failed to fetch data:", response.status_code, response.text)
        return None

    data = response.json()["list"][0]

    result = {
        "timestamp": datetime.utcfromtimestamp(data["dt"]),
        "aqi_index": data["main"]["aqi"],
        "co": data["components"]["co"],
        "no": data["components"]["no"],
        "no2": data["components"]["no2"],
        "o3": data["components"]["o3"],
        "so2": data["components"]["so2"],
        "pm2_5": data["components"]["pm2_5"],
        "pm10": data["components"]["pm10"],
        "nh3": data["components"]["nh3"]
    }

    return pd.DataFrame([result])

if __name__ == "__main__":
    df = fetch_air_pollution(LAT, LON, API_KEY)
    if df is not None:
        print("✅ Fetched AQI Data for Islamabad:")
        print(df)
    else:
        print("⚠️ No data received.")
