# train.py
"""
Training script for PPO residual agent.
This script uses MyEnv2 which now expects the agent to output residual corrections
in the range [-1, 1] for both velocity and omega.

Notes / small changes:
 - The environment action_space is Box([-1,-1],[1,1]) (residuals).
 - Agent logs/training settings can be tuned; recommended to start with small residual scale.
"""

import os
import time
from datetime import datetime

import gymnasium as gym
import pybullet as p
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

import leoroverpybullet  # ensure package registration
# Import the environment by its registration name (unchanged)
# Make sure the package __init__ registers the environment "leoroverpybullet/MyEnv2"

# --- PyBullet connection (use GUI for debugging or DIRECT for headless) ---
p.connect(p.GUI)

# Create environment
env = gym.make("leoroverpybullet/MyEnv2", display=True, debug=False)  # pass debug True for verbose prints
env = Monitor(env, filename="monitor.csv")

p.setTimeStep(1.0 / 50.0)

# Training config: continue or new
CONTINUE_TRAINING = False
MODEL_PATH = "models/ppo_leorover_model_latest.zip"

if CONTINUE_TRAINING and os.path.exists(MODEL_PATH):
    print(f"Loading existing model from: {MODEL_PATH}")
    agent = PPO.load(MODEL_PATH, env=env, device="cpu")
else:
    print("Creating a new model from scratch")
    agent = PPO(
        MlpPolicy,
        env,
        device="cpu",
        learning_rate=3e-4,
        n_steps=2048,
        clip_range=0.3,
        ent_coef=0.04,
        verbose=1,
    )

# Logging and checkpointing
now = datetime.now()
format_now = now.strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"./sb3_logs_{format_now}/"
os.makedirs("models", exist_ok=True)

new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
agent.set_logger(new_logger)

check_callback = CheckpointCallback(
    save_freq=20_000,
    save_path='models/',
    name_prefix='ppo_leorover_model',
)

# Train
agent.learn(
    total_timesteps=10_000_000,
    callback=check_callback,
    progress_bar=True
)

# Save final model & latest pointer
final_model_path = f"leoroverpybullet/models/ppo_leorover_model_{format_now}.zip"
agent.save(final_model_path)
agent.save(MODEL_PATH)
print(f"Saved final model to: {final_model_path}")

p.disconnect()
