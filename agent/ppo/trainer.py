from typing import Callable
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def env_factory() -> gym.Env:
    raise Exception("Not implemented")

model = PPO("MlpPolicy", env_factory(), verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")
