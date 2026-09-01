import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_PATH = "data/raw/raw_grid_weather.parquet"
OUTPUT_PATH = "data/processed/processed_grid_features.parquet"

class GridFeatureEngineer:
    """
    Cleans raw grid and weather telemetry, generates net load metrics,
    encodes cyclic time embeddings, and constructs rolling lag features.
    """

    def __init__(self, input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH):
        self.input_path = input_path
        self.output_path = output_path

    def load_data(self) -> pd.DataFrame:
        """Loads raw dataset from Parquet."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(
                f"Raw data file not found at '{self.input_path}'. "
                "Ensure Phase 2 ingestion script has been run."
            )
        logging.info(f"Loading raw data from '{self.input_path}'...")
        return pd.read_parquet(self.input_path)

    def clean_missing_and_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills timestamp gaps, interpolates missing values, and caps extreme outliers."""
        logging.info("Cleaning raw data and enforcing continuous hourly frequency...")
        
        # Ensure timestamp ordering
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.sort_values("timestamp", inplace=True)
        
        # Set timestamp index to reindex missing hours if any exist
        df.set_index("timestamp", inplace=True)
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h", tz="UTC")
        df = df.reindex(full_idx)
        df.index.name = "timestamp"

        # Time-based linear interpolation for short missing windows
        df = df.interpolate(method="time", limit=3)
        
        # Forward/backward fill for remaining boundary gaps
        df = df.ffill().bfill()
        df.reset_index(inplace=True)

        # Handle negative or zero anomalies in solar/demand metrics
        if "Net Generation - Solar" in df.columns:
            df["Net Generation - Solar"] = np.maximum(0, df["Net Generation - Solar"])
        if "Net Generation - Wind" in df.columns:
            df["Net Generation - Wind"] = np.maximum(0, df["Net Generation - Wind"])

        return df

    def compute_net_load(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Net Load (Demand minus Solar and Wind generation)."""
        logging.info("Computing Net Load metric...")
        
        demand = df.get("Demand", 0.0)
        solar = df.get("Net Generation - Solar", 0.0)
        wind = df.get("Net Generation - Wind", 0.0)
        
        df["net_load"] = demand - (solar + wind)
        df["renewable_ratio"] = np.where(demand > 0, (solar + wind) / demand, 0.0)
        df["renewable_ratio"] = np.clip(df["renewable_ratio"], 0.0, 1.0)
        
        return df

    def generate_cyclic_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes calendar temporal markers into continuous sine/cosine values."""
        logging.info("Building cyclic temporal embeddings (Hour, Day of Week, Day of Year)...")
        
        dt = df["timestamp"].dt
        
        # Hour of Day (24-hour cycle)
        hour = dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)

        # Day of Week (7-day cycle)
        day_of_week = dt.dayofweek
        df["day_sin"] = np.sin(2 * np.pi * day_of_week / 7.0).astype(np.float32)
        df["day_cos"] = np.cos(2 * np.pi * day_of_week / 7.0).astype(np.float32)

        # Day of Year (365-day annual cycle)
        day_of_year = dt.dayofyear
        df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25).astype(np.float32)
        df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25).astype(np.float32)

        # Categorical time indicators for TFT static categoricals
        df["is_weekend"] = (day_of_week >= 5).astype(np.int32)
        
        return df

    def generate_lag_and_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates historical autoregressive lags and dynamic rolling statistics."""
        logging.info("Constructing historical lags (24h, 48h, 168h) and rolling window statistics...")

        # Core target columns to lag
        target_cols = ["net_load", "Locational Marginal Price"]
        lag_intervals = [1, 2, 24, 48, 168]  # 1h, 2h, 1 day, 2 days, 1 week

        for col in target_cols:
            if col in df.columns:
                safe_name = col.lower().replace(" ", "_")
                for lag in lag_intervals:
                    df[f"{safe_name}_lag_{lag}h"] = df[col].shift(lag).astype(np.float32)

        # Dynamic rolling metrics for weather and grid stability
        window_sizes = [3, 6, 24]
        for w in window_sizes:
            df[f"net_load_rolling_mean_{w}h"] = df["net_load"].rolling(window=w).mean().astype(np.float32)
            df[f"net_load_rolling_std_{w}h"] = df["net_load"].rolling(window=w).std().astype(np.float32)
            
            if "global_horizontal_irradiance" in df.columns:
                df[f"ghi_rolling_mean_{w}h"] = df["global_horizontal_irradiance"].rolling(window=w).mean().astype(np.float32)
            if "wind_speed_100m" in df.columns:
                df[f"wind_speed_rolling_std_{w}h"] = df["wind_speed_100m"].rolling(window=w).std().astype(np.float32)

        # Temperature ramp rate (Hourly change)
        if "temperature" in df.columns:
            df["temp_ramp_rate"] = df["temperature"].diff().astype(np.float32)

        return df

    def run_pipeline(self) -> pd.DataFrame:
        """Executes full feature engineering workflow and saves processed Parquet file."""
        df = self.load_data()
        df = self.clean_missing_and_outliers(df)
        df = self.compute_net_load(df)
        df = self.generate_cyclic_temporal_features(df)
        df = self.generate_lag_and_rolling_features(df)

        # Drop initial rows containing NaN values caused by 168-hour (1-week) lag windowing
        logging.info("Dropping initial warmup rows containing lag NaN values...")
        df_clean = df.dropna().reset_index(drop=True)

        # Export processed file
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df_clean.to_parquet(self.output_path, engine="pyarrow", index=False)

        logging.info(f"SUCCESS: Processed dataset saved to '{self.output_path}'")
        logging.info(f"Final Output Shape: {df_clean.shape} (Rows, Columns)")
        return df_clean

if __name__ == "__main__":
    engineer = GridFeatureEngineer()
    engineer.run_pipeline()