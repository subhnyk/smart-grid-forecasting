import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration
st.set_page_config(
    page_title="Grid Dispatch & Forecasting Dashboard",
    page_icon="⚡",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"
DATA_PATH = "data/processed/processed_grid_features.parquet"

st.title("⚡ Adaptive Energy Grid Load & Autonomous Dispatch Dashboard")
st.markdown("Real-time TFT Multi-Horizon Load & Price Forecasting paired with PPO Battery Arbitrage Control.")

# --- Sidebar Controls ---
st.sidebar.header("🕹️ Simulation Controls")

battery_capacity_mwh = st.sidebar.number_input("Battery Capacity (MWh)", value=40.0, step=5.0)
max_power_mw = st.sidebar.number_input("Max Battery Power (MW)", value=10.0, step=1.0)
current_soc = st.sidebar.slider("Current State of Charge (SoC)", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("📡 API Microservice Connection")

# Check FastAPI Health State
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=3)
    if health_resp.status_code == 200:
        st.sidebar.success("FastAPI Service: ONLINE")
        api_data = health_resp.json()
        st.sidebar.caption(f"TFT Model: {'Loaded' if api_data.get('tft_loaded') else 'Fallback'}")
        st.sidebar.caption(f"PPO Agent: {'Loaded' if api_data.get('ppo_loaded') else 'Rule Policy'}")
    else:
        st.sidebar.error("FastAPI Service: UNHEALTHY")
except Exception:
    st.sidebar.error("FastAPI Service: OFFLINE (Run uvicorn src.serving.app:app)")

# --- Data Loading ---
@st.cache_data
def load_historical_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at '{DATA_PATH}'. Ensure Phase 3 has been run.")
        return None
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

df_grid = load_historical_data()

if df_grid is not None:
    st.sidebar.markdown("---")
    sim_index = st.sidebar.slider(
        "Simulation Sequence Index",
        min_value=24,
        max_value=len(df_grid) - 25,
        value=100
    )

    past_24h_df = df_grid.iloc[sim_index - 24:sim_index].copy()

    # Build Telemetry Payload for FastAPI app.py
    telemetry_payload = []
    for _, row in past_24h_df.iterrows():
        telemetry_payload.append({
            "timestamp": row["timestamp"].isoformat(),
            "demand_mw": float(row.get("Demand", 4000.0)),
            "solar_mw": float(row.get("Net Generation - Solar", 0.0)),
            "wind_mw": float(row.get("Net Generation - Wind", 0.0)),
            "price_dollar_mwh": float(row.get("Locational Marginal Price", 35.0)),
            "temperature_c": float(row.get("temperature", 25.0)),
            "ghi_w_m2": float(row.get("global_horizontal_irradiance", 0.0)),
            "wind_speed_m_s": float(row.get("wind_speed_100m", 5.0))
        })

    api_request_data = {
        "current_soc": current_soc,
        "historical_telemetry": telemetry_payload
    }

    if st.button("🚀 Trigger Real-Time Inference & Battery Dispatch", use_container_width=True):
        try:
            with st.spinner("Querying TFT Multi-Horizon Model & Executing PPO Policy..."):
                response = requests.post(f"{API_URL}/predict/dispatch", json=api_request_data, timeout=10)
                
            if response.status_code == 200:
                result = response.json()
                
                # KPI Summary Cards
                col1, col2, col3, col4 = st.columns(4)
                
                action_type = result["action_type"]
                action_color = "🟢" if action_type == "CHARGE" else ("🔴" if action_type == "DISCHARGE" else "⚪")
                
                col1.metric("Recommended Action", f"{action_color} {action_type}")
                col2.metric("Dispatch Power", f"{result['action_power_mw']} MW")
                col3.metric("Projected State of Charge", f"{result['target_soc'] * 100:.1f} %")
                col4.metric("Est. Financial Flow", f"${result['estimated_revenue_dollar']:,.2f}")
                
                st.markdown("---")

                # TFT Quantile Forecast Plots
                df_forecast = pd.DataFrame(result["forecast_next_24h"])
                st.subheader("📈 TFT Multi-Horizon Net Load Prediction Bounds (Next 24 Hours)")
                
                fig_forecast = go.Figure()

                fig_forecast.add_trace(go.Scatter(
                    x=df_forecast["timestep_ahead"], y=df_forecast["net_load_p90"],
                    mode="lines", line=dict(width=0), showlegend=False, name="P90"
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=df_forecast["timestep_ahead"], y=df_forecast["net_load_p10"],
                    mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(0, 150, 255, 0.2)", name="Quantile Confidence Interval (P10-P90)"
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=df_forecast["timestep_ahead"], y=df_forecast["net_load_p50"],
                    mode="lines+markers", line=dict(color="#0066FF", width=3), name="Median Forecast (P50 Net Load)"
                ))

                fig_forecast.update_layout(
                    xaxis_title="Timestep Ahead (Hours)", yaxis_title="Net Load (MW)",
                    hovermode="x unified", template="plotly_dark", height=400
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

                # Contextual Historical Plot
                st.subheader("📊 Past 24 Hours Context & Market Dynamics")
                fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
                fig_hist.add_trace(
                    go.Bar(x=past_24h_df["timestamp"], y=past_24h_df["net_load"], name="Historical Net Load (MW)", marker_color="#2E86C1", opacity=0.7),
                    secondary_y=False
                )
                fig_hist.add_trace(
                    go.Scatter(x=past_24h_df["timestamp"], y=past_24h_df["Locational Marginal Price"], name="LMP ($/MWh)", line=dict(color="#E74C3C", width=2.5)),
                    secondary_y=True
                )
                fig_hist.update_layout(template="plotly_dark", height=400, hovermode="x unified")
                st.plotly_chart(fig_hist, use_container_width=True)

            else:
                st.error(f"API Error Response: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Failed to communicate with FastAPI microservice: {e}")