import os
import logging
import pandas as pd
from src.ingestion.eia_client import EIAv2Client
from src.ingestion.weather_client import OpenMeteoClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OUTPUT_PATH = "data/raw/raw_grid_weather.parquet"

def execute_pipeline():
    
    # 1. Initialize Clients
    eia = EIAv2Client()
    weather = OpenMeteoClient(latitude=29.7604, longitude=-95.3698)

    # 2. Fetch Time Series Data
    df_grid = eia.fetch_hourly_grid_data(start_date="2025-01-01T00", end_date="2025-12-31T23")
    df_weather = weather.fetch_hourly_weather(start_date="2025-01-01", end_date="2025-12-31")

    # 3. Merge datasets on UTC Timestamps
    logging.info("Merging EIA Grid telemetry and Open-Meteo Weather data...")
    df_merged = pd.merge(df_grid, df_weather, on="timestamp", how="inner")

    # Sort sequentially by timestamp
    df_merged.sort_values("timestamp", inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    # 4. Save to Disk in Parquet Format
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_merged.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False)

    logging.info(f"SUCCESS: Raw dataset saved to '{OUTPUT_PATH}'")
    logging.info(f"Merged Dataset Shape: {df_merged.shape}")
    logging.info("Sample Columns: " + ", ".join(df_merged.columns.tolist()))
    

if __name__ == "__main__":
    execute_pipeline()
    