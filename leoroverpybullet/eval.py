import time
import gymnasium as gym
import pybullet as p
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
import leoroverpybullet

p.connect(p.GUI)
env = gym.make("leoroverpybullet/MyEnv2", display=True)
env = Monitor(env, filename="monitor_eval.csv")
p.setTimeStep(1.0 / 50)
agent = PPO.load("models/ppo_leorover_model_latest.zip")
obs, _ = env.reset()
done = False
start_time = time.time()
while not done:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    print(f"time: {time.time() - start_time:.2f}, reward: {reward:.2f}")
p.disconnect()