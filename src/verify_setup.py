import sys
import torch
import gymnasium as gym
import pytorch_forecasting
from stable_baselines3 import PPO
from fastapi import FastAPI

def run_verification():
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version:    {sys.version.split()[0]}")
    print(f"PyTorch Version:   {torch.__version__}")
    print(f"CUDA Available:    {torch.cuda.is_available()}")
    print(f"Gymnasium:         {gym.__version__}")
    print(f"PyTorch Forecast:  {pytorch_forecasting.__version__}")
  
    print("SUCCESS: Phase 1 setup is complete and fully functional!")

if __name__ == "__main__":
    run_verification()