import os
import logging
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EIAv2Client:
    """
    Client for fetching hourly electric grid operations data from the US EIA v2 API.
    Docs: https://www.eia.gov/opendata/documentation.php
    """

    BASE_URL = "https://api.eia.gov/v2/electricity/rto/interchange/data/"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("EIA_API_KEY")
        if not self.api_key or self.api_key == "your_actual_eia_v2_api_key_here":
            logging.warning("EIA_API_KEY is not set or using placeholder. Running EIA client in fallback mode.")

    def fetch_hourly_grid_data(
        self,
        respondent: str = "ERCO",  # ERCOT (Texas) regional code
        start_date: str = "2025-01-01T00",
        end_date: str = "2025-12-31T23",
    ) -> pd.DataFrame:
        """
        Extracts hourly demand, solar generation, and wind generation with pagination.
        """
        if not self.api_key or self.api_key == "your_actual_eia_v2_api_key_here":
            return self._generate_fallback_eia_data(start_date, end_date)

        endpoint = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
        headers = {"Accept": "application/json"}

        all_records = []
        offset = 0
        length = 5000

        while True:
            params = {
                "api_key": self.api_key,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": respondent,
                "start": start_date,
                "end": end_date,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": length,
            }

            logging.info(f"Querying EIA v2 API (offset={offset}) for region: {respondent}...")
            try:
                response = requests.get(endpoint, headers=headers, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                records = data.get("response", {}).get("data", [])
                if not records:
                    break

                all_records.extend(records)

                # Check if we've reached the end of the data
                if len(records) < length:
                    break

                offset += length
            except Exception as e:
                logging.error(f"Error during API pagination: {e}")
                break

        if not all_records:
            logging.warning("EIA API returned no records. Switching to fallback generator.")
            return self._generate_fallback_eia_data(start_date, end_date)

        df = pd.DataFrame(all_records)

        # Pivot EIA response to have distinct columns per metric type
        df_pivot = df.pivot_table(
            index="period",
            columns="type-name",
            values="value",
            aggfunc="first"
        ).reset_index()

        df_pivot.rename(columns={"period": "timestamp"}, inplace=True)
        df_pivot["timestamp"] = pd.to_datetime(df_pivot["timestamp"], utc=True)

        logging.info(f"Successfully fetched {len(df_pivot)} EIA grid records.")
        return df_pivot

    def _generate_fallback_eia_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generates realistic synthetic EIA grid data based on provided dates."""
        logging.info(f"Generating synthetic fallback EIA operational data from {start_date} to {end_date}...")

        # Parse dates for range generation
        try:
            start_dt = pd.to_datetime(start_date).tz_localize('UTC') if not 'T' in start_date else pd.to_datetime(start_date + " :00").tz_localize('UTC')
            end_dt = pd.to_datetime(end_date).tz_localize('UTC') if not 'T' in end_date else pd.to_datetime(end_date + " :00").tz_localize('UTC')
            dates = pd.date_range(start=start_dt, end=end_dt, freq="h", tz="UTC")
        except Exception:
            # Fallback to a default year if parsing fails
            dates = pd.date_range(start="2025-01-01", end="2025-12-31 23:00:00", freq="h", tz="UTC")

        hours = dates.hour
        days = dates.dayofyear

        np.random.seed(42)

        total_load = 22000 + 4000 * np.sin(2 * np.pi * hours / 24) + 3000 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 400, len(dates))
        solar_gen = np.maximum(0, 7000 * np.sin(np.pi * (hours - 6) / 12)) * ((hours >= 6) & (hours <= 18)) + np.random.normal(0, 100, len(dates))
        wind_gen = 6000 + 2000 * np.cos(2 * np.pi * hours / 24) + np.random.normal(0, 500, len(dates))
        spot_price = 25.0 + 0.0000005 * (total_load ** 2) - 0.001 * (wind_gen + solar_gen) + np.random.normal(0, 3, len(dates))

        df = pd.DataFrame({
            "timestamp": dates,
            "Demand": np.clip(total_load, 10000, None).astype(np.float32),
            "Net Generation - Solar": np.clip(solar_gen, 0, None).astype(np.float32),
            "Net Generation - Wind": np.clip(wind_gen, 0, None).astype(np.float32),
            "Locational Marginal Price": np.clip(spot_price, -10, 1000).astype(np.float32)
        })
        return df