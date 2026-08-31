import logging
import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class OpenMeteoClient:
    """
    Client for extracting hourly historical weather metrics from Open-Meteo ERA5 API.
    Docs: https://open-meteo.com/en/docs/historical-weather-api
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, latitude: float = 29.7604, longitude: float = -95.3698):
        # Default coordinates set to Houston / ERCOT Coastal Region
        self.latitude = latitude
        self.longitude = longitude

    def fetch_hourly_weather(
        self,
        start_date: str = "2025-01-01",
        end_date: str = "2025-12-31"
    ) -> pd.DataFrame:
        """
        Extracts temperature, solar irradiance (GHI/DNI), and 100m wind speeds.
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m",
                "direct_normal_irradiance",
                "global_horizontal_irradiance",
                "wind_speed_100m",
                "surface_pressure"
            ],
            "timezone": "UTC"
        }

        logging.info(f"Querying Open-Meteo API for coordinates ({self.latitude}, {self.longitude})...")

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            hourly_data = data.get("hourly", {})
            if not hourly_data:
                logging.warning("Open-Meteo returned empty payload. Using synthetic fallback.")
                return self._generate_fallback_weather(start_date, end_date)

            df = pd.DataFrame({
                "timestamp": pd.to_datetime(hourly_data["time"], utc=True),
                "temperature": np.array(hourly_data["temperature_2m"], dtype=np.float32),
                "direct_normal_irradiance": np.array(hourly_data["direct_normal_irradiance"], dtype=np.float32),
                "global_horizontal_irradiance": np.array(hourly_data["global_horizontal_irradiance"], dtype=np.float32),
                "wind_speed_100m": np.array(hourly_data["wind_speed_100m"], dtype=np.float32),
                "surface_pressure": np.array(hourly_data["surface_pressure"], dtype=np.float32)
            })

            logging.info(f"Successfully fetched {len(df)} weather telemetry records.")
            return df

        except Exception as e:
            logging.error(f"Failed to fetch Open-Meteo weather data: {e}. Generating fallback dataset.")
            return self._generate_fallback_weather(start_date, end_date)

    def _generate_fallback_weather(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generates synthetic weather telemetry if API fails."""
        logging.info("Generating synthetic fallback Open-Meteo weather data...")
        dates = pd.date_range(start=start_date, end=f"{end_date} 23:00:00", freq="h", tz="UTC")
        hours = dates.hour
        days = dates.dayofyear

        temp = 18 + 10 * np.sin(2 * np.pi * days / 365) + 4 * np.cos(2 * np.pi * hours / 24) + np.random.normal(0, 1.5, len(dates))
        dni = np.maximum(0, 800 * np.sin(np.pi * (hours - 6) / 12)) * ((hours >= 6) & (hours <= 18))
        ghi = np.maximum(0, 600 * np.sin(np.pi * (hours - 6) / 12)) * ((hours >= 6) & (hours <= 18))
        wind = 7 + 3 * np.sin(2 * np.pi * hours / 12) + np.random.normal(0, 1.5, len(dates))
        pressure = 1013.25 + np.random.normal(0, 4, len(dates))

        return pd.DataFrame({
            "timestamp": dates,
            "temperature": temp.astype(np.float32),
            "direct_normal_irradiance": dni.astype(np.float32),
            "global_horizontal_irradiance": ghi.astype(np.float32),
            "wind_speed_100m": np.clip(wind, 0, None).astype(np.float32),
            "surface_pressure": pressure.astype(np.float32)
        })