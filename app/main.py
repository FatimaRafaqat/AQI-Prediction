from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import hopsworks
import os
import pandas as pd

load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

# Connect to Hopsworks
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT"),
    host=os.getenv("HOPSWORKS_HOST")
)

fs = project.get_feature_store()

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    try:
        # Fetch predictions from Feature Store
        feature_group = fs.get_feature_group(name="model_predictions", version=1)
        df = feature_group.read()

        # Convert prediction_ts to datetime
        df['prediction_ts'] = pd.to_datetime(df['prediction_ts'], errors='coerce')

        # Get latest timestamp
        latest_ts = df['prediction_ts'].max()

        # Filter only the latest run
        df = df[df['prediction_ts'] == latest_ts]

        # Sort by horizon_day
        df = df.sort_values(by="horizon_day")

        # Format date for display
        df['prediction_date'] = pd.to_datetime(df['prediction_date']).dt.strftime('%Y-%m-%d')

        # Convert to list of dicts for HTML
        predictions = df.to_dict(orient="records")

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "predictions": predictions, "error": None}
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "predictions": None, "error": str(e)}
        )
