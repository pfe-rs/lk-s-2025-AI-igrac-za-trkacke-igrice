from dataclasses import dataclass
from pathlib import Path
import argparse


@dataclass
class BaseTrainerArgs:
    model_path: Path
    levels_path: Path
    training_log_path: Path
    checkpoints_path: Path
    best_path: Path
    total_timesteps: int
    epochs: int  # Added here


class BaseTrainer:
    DEFAULT_MODEL_PATH = Path("./models/model")
    DEFAULT_LEVELS_PATH = Path("./levels")
    DEFAULT_LOG_PATH = Path("./logs/training")
    DEFAULT_CHECKPOINTS_PATH = Path("./models/checkpoints")
    DEFAULT_BEST_PATH = Path("./models/best")
    DEFAULT_TOTAL_TIMESTEPS = 100000
    DEFAULT_EPOCHS = 100

    def __init__(self):
        self.args_parser = argparse.ArgumentParser(description="Train a model for a racing game.")
        
        self.args_parser.add_argument(
            "--model_path", type=Path, default=self.DEFAULT_MODEL_PATH,
            help=f"Path to save/load the PPO model (default: {self.DEFAULT_MODEL_PATH})"
        )
        self.args_parser.add_argument(
            "--levels_path", type=Path, default=self.DEFAULT_LEVELS_PATH,
            help=f"Path to the directory containing environment levels (default: {self.DEFAULT_LEVELS_PATH})"
        )
        self.args_parser.add_argument(
            "--checkpoints_path", type=Path, default=self.DEFAULT_CHECKPOINTS_PATH,
            help=f"Path to save training checkpoints (default: {self.DEFAULT_CHECKPOINTS_PATH})"
        )
        self.args_parser.add_argument(
            "--log_path", type=Path, default=self.DEFAULT_LOG_PATH,
            help=f"Path to save training logs (default: {self.DEFAULT_LOG_PATH})"
        )
        self.args_parser.add_argument(
            "--best_path", type=Path, default=self.DEFAULT_BEST_PATH,
            help=f"Path to save best models (default: {self.DEFAULT_BEST_PATH})"
        )
        self.args_parser.add_argument(
            "--total_timesteps", type=int, default=self.DEFAULT_TOTAL_TIMESTEPS,
            help=f"Total number of timesteps to train (default: {self.DEFAULT_TOTAL_TIMESTEPS})"
        )
        self.args_parser.add_argument(
            "--epochs", type=int, default=self.DEFAULT_EPOCHS,
            help=f"Optional limit for number of epochs (default: {self.DEFAULT_EPOCHS})"
        )

    def _parse_args(self) -> BaseTrainerArgs:
        args = self.args_parser.parse_args()
        return BaseTrainerArgs(
            model_path=args.model_path,
            levels_path=args.levels_path,
            training_log_path=args.log_path,
            checkpoints_path=args.checkpoints_path,
            best_path=args.best_path,
            total_timesteps=args.total_timesteps,
            epochs=args.epochs
        )
