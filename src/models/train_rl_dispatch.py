import os
import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_PATH = "data/processed/processed_grid_features.parquet"
MODEL_SAVE_DIR = "models/"
RL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "ppo_battery_dispatch.zip")

class BatteryDispatchEnv(gym.Env):
    """
    Custom Gymnasium Environment for Battery Storage Arbitrage and Load Leveling.
    Simulates a 10MW / 40MWh Grid-Scale Lithium-Ion Battery Energy Storage System.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        capacity_mwh: float = 40.0,
        max_power_mw: float = 10.0,
        roundtrip_efficiency: float = 0.90,
        degradation_cost_per_mwh: float = 5.0,
    ):
        super(BatteryDispatchEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.capacity_mwh = capacity_mwh
        self.max_power_mw = max_power_mw
        self.eta_charge = np.sqrt(roundtrip_efficiency)      # ~0.9487
        self.eta_discharge = np.sqrt(roundtrip_efficiency)   # ~0.9487
        self.degradation_cost = degradation_cost_per_mwh

        # Min and max SOC bounds (10% to 90% usable energy window to prevent deep degradation)
        self.min_soc = 0.10
        self.max_soc = 0.90

        # Normalization scale factors
        self.price_max = float(self.df["Locational Marginal Price"].max()) + 1e-5
        self.net_load_max = float(self.df["net_load"].max()) + 1e-5

        # Action Space: [-1.0, 1.0] -> -1.0 Max Discharge, +1.0 Max Charge
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space: [SoC, Price, Net_Load, Price_t+1, Hour_sin, Hour_cos]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.current_step = 0
        self.soc = 0.50  # Start at 50% SoC

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        next_row = self.df.iloc[min(self.current_step + 1, len(self.df) - 1)]

        price_norm = row["Locational Marginal Price"] / self.price_max
        net_load_norm = row["net_load"] / self.net_load_max
        next_price_norm = next_row["Locational Marginal Price"] / self.price_max

        return np.array([
            self.soc,
            price_norm,
            net_load_norm,
            next_price_norm,
            row["hour_sin"],
            row["hour_cos"]
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.soc = 0.50  # Reset to 50% charge state
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        act = float(np.clip(action[0], -1.0, 1.0))
        row = self.df.iloc[self.current_step]
        price = float(row["Locational Marginal Price"])  # $/MWh

        energy_delta = 0.0
        actual_power_mw = 0.0
        penalty = 0.0

        if act > 0:
            # Charging action
            requested_power = act * self.max_power_mw
            energy_needed = requested_power * self.eta_charge  # MWh added to battery
            max_addable_soc = self.max_soc - self.soc
            max_addable_mwh = max_addable_soc * self.capacity_mwh

            if energy_needed > max_addable_mwh:
                # Overcharge attempt: Cap charge power and assess penalty
                actual_mwh_added = max_addable_mwh
                actual_power_mw = actual_mwh_added / self.eta_charge
                penalty += 10.0 * (energy_needed - max_addable_mwh)
            else:
                actual_mwh_added = energy_needed
                actual_power_mw = requested_power

            self.soc += actual_mwh_added / self.capacity_mwh
            # Cost paid to grid to charge
            financial_flow = -(actual_power_mw * price)

        elif act < 0:
            # Discharging action
            requested_power = abs(act) * self.max_power_mw
            energy_from_battery = requested_power / self.eta_discharge
            max_drawable_soc = self.soc - self.min_soc
            max_drawable_mwh = max_drawable_soc * self.capacity_mwh

            if energy_from_battery > max_drawable_mwh:
                # Deep discharge attempt: Cap power and assess penalty
                actual_mwh_drawn = max_drawable_mwh
                actual_power_mw = actual_mwh_drawn * self.eta_discharge
                penalty += 10.0 * (energy_from_battery - max_drawable_mwh)
            else:
                actual_mwh_drawn = energy_from_battery
                actual_power_mw = requested_power * self.eta_discharge

            self.soc -= actual_mwh_drawn / self.capacity_mwh
            # Revenue earned selling power to grid
            financial_flow = (actual_power_mw * price)

        else:
            # Idle action
            financial_flow = 0.0

        # Degradation cost per MWh throughput
        throughput_mwh = abs(actual_power_mw)
        degradation_loss = throughput_mwh * self.degradation_cost

        # Total Step Reward calculation
        reward = financial_flow - degradation_loss - penalty

        # Advance timestep
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        self.soc = float(np.clip(self.soc, self.min_soc, self.max_soc))
        obs = self._get_obs()

        info = {
            "step": self.current_step,
            "financial_flow": financial_flow,
            "soc": self.soc,
            "price": price
        }

        return obs, reward, terminated, truncated, info


def train_rl_agent(total_timesteps: int = 100_000):
    """Loads features, constructs environment, and runs Stable-Baselines3 PPO training."""
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Processed dataset not found at '{INPUT_PATH}'. Complete Phase 3 first.")

    logging.info("Loading dataset into RL Dispatch Environment...")
    df = pd.read_parquet(INPUT_PATH)

    # Instantiate vector environment wrapper
    env_fn = lambda: BatteryDispatchEnv(df=df)
    env = DummyVecEnv([env_fn])

    logging.info("Initializing PPO Neural Network Policy...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="tb_logs/ppo_battery/"
    )

    logging.info(f"Training PPO Agent for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    # Save trained model weights
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model.save(RL_MODEL_PATH)
    logging.info(f"SUCCESS: Trained PPO model saved to '{RL_MODEL_PATH}'")

if __name__ == "__main__":
    train_rl_agent(total_timesteps=100_000)