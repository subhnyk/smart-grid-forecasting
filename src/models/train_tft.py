import os
import logging
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.metrics import QuantileLoss, RMSE, MAE
from pytorch_forecasting.data import GroupNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Fix random seed for reproducibility
pl.seed_everything(42)

INPUT_PATH = "data/processed/processed_grid_features.parquet"
MODEL_SAVE_DIR = "models/"
TFT_CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "tft_model.ckpt")

class TFTGridForecaster:
    """
    Temporal Fusion Transformer pipeline for joint 24-hour ahead quantile 
    forecasting of Net Load and Locational Marginal Price (LMP).
    """

    def __init__(self, data_path: str = INPUT_PATH, max_encoder_length: int = 24, max_prediction_length: int = 24):
        self.data_path = data_path
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length

    def load_and_prepare_dataset(self):
        """Loads processed data, adds sequence indices, and builds TimeSeriesDataSet."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Processed dataset not found at '{self.data_path}'. Complete Phase 3 first.")

        logging.info("Loading feature-engineered dataset for TFT training...")
        df = pd.read_parquet(self.data_path)

        # TFT requires an integer time index for ordering
        df["time_idx"] = np.arange(len(df))
        
        # Artificial static group identifier (single grid region, e.g., ERCOT)
        df["region"] = "ERCOT_GRID"
        
        # Enforce float32 data types across features
        float_cols = [
            "net_load", "Locational Marginal Price", "temperature",
            "global_horizontal_irradiance", "wind_speed_100m",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "day_of_year_sin", "day_of_year_cos",
            "net_load_lag_24h", "locational_marginal_price_lag_24h",
            "net_load_rolling_mean_24h", "net_load_rolling_std_24h"
        ]
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        # Train/Validation Cutoff (Time-based split: 85% train, 15% validation)
        training_cutoff = df["time_idx"].max() - int(len(df) * 0.15)

        logging.info(f"Total Timesteps: {len(df)} | Training Cutoff Index: {training_cutoff}")

        # Construct PyTorch Forecasting TimeSeriesDataSet
        training_dataset = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="net_load",
            group_ids=["region"],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["region"],
            time_varying_known_reals=[
                "time_idx", "hour_sin", "hour_cos", 
                "day_sin", "day_cos", "global_horizontal_irradiance", 
                "wind_speed_100m", "temperature"
            ],
            time_varying_unknown_reals=[
                "net_load", "Locational Marginal Price", 
                "net_load_lag_24h", "locational_marginal_price_lag_24h",
                "net_load_rolling_mean_24h", "net_load_rolling_std_24h"
            ],
            target_normalizer=GroupNormalizer(groups=["region"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, 
            df, 
            predict=True, 
            stop_randomization=True
        )

        return training_dataset, validation_dataset, df

    def train(self, max_epochs: int = 15, batch_size: int = 64):
        """Builds DataLoaders, constructs TFT architecture, and runs PyTorch Lightning trainer."""
        train_ds, val_ds, raw_df = self.load_and_prepare_dataset()

        train_dataloader = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = val_ds.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

        # Evaluate Baseline Model (Persistence baseline)
        baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
        baseline_mae = MAE()(baseline_predictions.output, baseline_predictions.y).item()
        logging.info(f"Baseline (Naive Persistence) Validation MAE: {baseline_mae:.4f}")

        # Construct Temporal Fusion Transformer
        tft = TemporalFusionTransformer.from_dataset(
            train_ds,
            learning_rate=1e-3,
            hidden_size=32,          # Dynamic network capacity
            attention_head_size=2,   # Multi-head self-attention
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=7,           # 7 Quantiles for QuantileLoss ([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
            loss=QuantileLoss([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]),
            reduce_on_plateau_patience=4,
        )

        logging.info(f"TFT Network Architecture Initialized. Parameter Count: {sum(p.numel() for p in tft.parameters()):,}")

        # Setup Lightning Callbacks & Loggers
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")
        checkpoint_callback = ModelCheckpoint(
            dirpath=MODEL_SAVE_DIR, 
            filename="tft_model", 
            monitor="val_loss", 
            mode="min",
            save_top_k=1
        )
        logger = TensorBoardLogger("tb_logs", name="tft_grid_forecaster")

        # PyTorch Lightning Trainer Execution
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=logger
        )

        logging.info("Starting TFT Training Epochs...")
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        logging.info(f"TFT Model training complete. Best Checkpoint Path: {checkpoint_callback.best_model_path}")
        return checkpoint_callback.best_model_path

if __name__ == "__main__":
    forecaster = TFTGridForecaster()
    forecaster.train(max_epochs=10, batch_size=64)
    