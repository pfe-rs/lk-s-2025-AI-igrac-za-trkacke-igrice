from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from pathlib import Path
import sys

from agent.ppo.env import env_factory

default_model_path = Path("./models/ppo_model")
default_level_path = Path("./levels")

if __name__ == "__main__":
    # Load paths from CLI args
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_model_path
    levels_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_level_path

    env = env_factory(levels_path)

    if model_path.exists():
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env, device="cpu")
    else:
        print(f"Training new model at {model_path}")
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")

    try:
        model.learn(total_timesteps=2500000000000)
        model.save(model_path)
    except KeyboardInterrupt:
        model.save(model_path)
