import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from agent.env import Env

default_model_path = "ppo_model"

def env_factory() -> gym.Env:
    raise Exception("Not implemented")

if __name__ == "__main__":
    model_path = default_model_path

    model = PPO("MlpPolicy", env_factory(), verbose=1)
    model.learn(total_timesteps=25000)
    model.save(model_path)

