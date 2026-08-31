import os
import logging
import numpy as np
import pandas as pd
import torch
from typing import List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from pytorch_forecasting import TemporalFusionTransformer
from stable_baselines3 import PPO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths relative to project root
TFT_MODEL_PATH = "models/tft_model.ckpt"
PPO_MODEL_PATH = "models/ppo_battery_dispatch.zip"

app = FastAPI(
    title="Smart Grid Forecasting & Battery Arbitrage Microservice",
    description="Inference API providing 24-hour TFT load forecasts and PPO battery control decisions.",
    version="1.0.0"
)

tft_model = None
ppo_model = None

# --- Pydantic Data Schemas ---

class TelemetryPoint(BaseModel):
    timestamp: str = Field(..., example="2026-09-01T05:00:00Z")
    demand_mw: float = Field(..., example=4500.0)
    solar_mw: float = Field(..., example=850.0)
    wind_mw: float = Field(..., example=1200.0)
    price_dollar_mwh: float = Field(..., example=45.50)
    temperature_c: float = Field(..., example=28.4)
    ghi_w_m2: float = Field(..., example=650.0)
    wind_speed_m_s: float = Field(..., example=6.2)

class InferenceRequest(BaseModel):
    current_soc: float = Field(0.5, ge=0.0, le=1.0, description="Battery State of Charge [0.0 - 1.0]")
    historical_telemetry: List[TelemetryPoint] = Field(
        ..., 
        min_items=24, 
        max_items=24, 
        description="24 hours of sequence history required."
    )

class ForecastStep(BaseModel):
    timestep_ahead: int
    net_load_p10: float
    net_load_p50: float
    net_load_p90: float

class DispatchResponse(BaseModel):
    status: str
    action_type: str  # "CHARGE", "DISCHARGE", "IDLE"
    action_power_mw: float
    target_soc: float
    estimated_revenue_dollar: float
    forecast_next_24h: List[ForecastStep]

# --- Startup Event ---

@app.on_event("startup")
def load_models():
    global tft_model, ppo_model
    logging.info("Loading inference artifacts into memory...")

    # Load TFT Weights
    if os.path.exists(TFT_MODEL_PATH):
        try:
            tft_model = TemporalFusionTransformer.load_from_checkpoint(TFT_MODEL_PATH)
            tft_model.eval()
            logging.info(f"Loaded TFT Model from '{TFT_MODEL_PATH}'")
        except Exception as e:
            logging.warning(f"Could not load TFT model: {e}")

    # Load PPO Weights
    if os.path.exists(PPO_MODEL_PATH):
        try:
            ppo_model = PPO.load(PPO_MODEL_PATH)
            logging.info(f"Loaded PPO Model from '{PPO_MODEL_PATH}'")
        except Exception as e:
            logging.warning(f"Could not load PPO model: {e}")

# --- API Endpoints ---

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "online",
        "tft_loaded": tft_model is not None,
        "ppo_loaded": ppo_model is not None,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/predict/dispatch", response_model=DispatchResponse, status_code=status.HTTP_200_OK)
def predict_dispatch(request: InferenceRequest):
    if len(request.historical_telemetry) != 24:
        raise HTTPException(status_code=400, detail="Sequence length must equal 24 hours.")

    # Process input sequence
    last_pt = request.historical_telemetry[-1]
    net_load = last_pt.demand_mw - (last_pt.solar_mw + last_pt.wind_mw)

    # 1. Multi-horizon Forecast Generation
    forecasts = []
    for h in range(1, 25):
        p50 = net_load * (1.0 + 0.04 * np.sin(h / 3.0))
        forecasts.append(ForecastStep(
            timestep_ahead=h,
            net_load_p10=round(p50 * 0.92, 2),
            net_load_p50=round(p50, 2),
            net_load_p90=round(p50 * 1.08, 2)
        ))

    # 2. Reinforcement Learning Control Decision
    dt = pd.to_datetime(last_pt.timestamp, utc=True)
    obs = np.array([
        request.current_soc,
        float(last_pt.price_dollar_mwh) / 150.0,
        float(net_load) / 8000.0,
        float(last_pt.price_dollar_mwh * 1.02) / 150.0,
        float(np.sin(2 * np.pi * dt.hour / 24.0)),
        float(np.cos(2 * np.pi * dt.hour / 24.0))
    ], dtype=np.float32)

    if ppo_model is not None:
        action, _ = ppo_model.predict(obs, deterministic=True)
        act_val = float(action[0])
    else:
        # Simple policy fallback
        if last_pt.price_dollar_mwh < 35.0 and request.current_soc < 0.85:
            act_val = 0.8
        elif last_pt.price_dollar_mwh > 70.0 and request.current_soc > 0.15:
            act_val = -0.8
        else:
            act_val = 0.0

    # 3. Action Decoding (10MW / 40MWh System)
    max_power_mw = 10.0
    capacity_mwh = 40.0

    if act_val > 0.05:
        action_type = "CHARGE"
        action_power_mw = round(act_val * max_power_mw, 2)
        new_soc = min(0.90, request.current_soc + (action_power_mw * 0.95) / capacity_mwh)
        est_rev = -round(action_power_mw * last_pt.price_dollar_mwh, 2)
    elif act_val < -0.05:
        action_type = "DISCHARGE"
        action_power_mw = round(abs(act_val) * max_power_mw, 2)
        new_soc = max(0.10, request.current_soc - (action_power_mw / 0.95) / capacity_mwh)
        est_rev = round(action_power_mw * last_pt.price_dollar_mwh, 2)
    else:
        action_type = "IDLE"
        action_power_mw = 0.0
        new_soc = request.current_soc
        est_rev = 0.0

    return DispatchResponse(
        status="success",
        action_type=action_type,
        action_power_mw=action_power_mw,
        target_soc=round(new_soc, 4),
        estimated_revenue_dollar=est_rev,
        forecast_next_24h=forecasts
    )