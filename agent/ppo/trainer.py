from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.ppo import PPO

from agent.base_trainer import BaseTrainer
from agent.ppo.env import env_factory


def get_latest_checkpoint(path: Path) -> Path | None:
    checkpoints = sorted(path.glob("ppo_*.zip"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


class PPOTrainer(BaseTrainer):
    DEFAULT_MODEL_PATH = Path("models/ppo_model")
    DEFAULT_CHECKPOINTS_PATH = Path("models/ppo_checkpoints")
    DEFAULT_BEST_PATH = Path("models/ppo_best")
    DEFAULT_LOG_PATH = Path("logs/ppo_training")
    model: PPO

    def __init__(self):
        super().__init__()
        self.args = self._parse_args()
        self.env = DummyVecEnv([lambda: env_factory(self.args.levels_path)])
        self.env = VecMonitor(self.env)
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        model_path = self.args.model_path

        last_checkpoint_path = get_latest_checkpoint(self.args.checkpoints_path)
        if last_checkpoint_path:
            print(f"Loading model from checkpoint {last_checkpoint_path}")
            self.model = PPO.load(last_checkpoint_path, self.env, device="cpu")
        else:
            if model_path.exists():
                print(f"Loading model from {model_path}")
                self.model = PPO.load(model_path, self.env, device="cpu")

            else:
                print(f"Training new model at {model_path}")
                self.model = PPO("MlpPolicy", self.env, verbose=1, device="cpu")

        self.model.ent_coef=0.03
        self.model.tensorboard_log = str(self.args.training_log_path)
        self._setup_callbacks()
        self._setup_logs()
        
    def _setup_logs(self):
        logger = configure(str(self.args.training_log_path), ["stdout", "json", "tensorboard"])
        self.model.set_logger(logger)

    def _setup_callbacks(self):
        checkpoint_cb = CheckpointCallback(
            save_freq=5_000,
            save_path=str(self.args.checkpoints_path),
            name_prefix='ppo',
            save_replay_buffer=True,
            save_vecnormalize=True,
            verbose=1
        )

        eval_cb = EvalCallback(
            self.env,
            best_model_save_path=str(self.args.best_path),
            eval_freq=5_000,
            verbose=1,
            deterministic=False
        )
        self.callbacks = CallbackList([
            checkpoint_cb,
            eval_cb
        ])
    
    def train(self):
        try:
            self.model.learn(
                total_timesteps=self.args.total_timesteps,
                reset_num_timesteps=False,
                callback=self.callbacks,
                tb_log_name="PPO",
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            raise KeyboardInterrupt
        finally:
            print("Saving model")
            self.model.save(self.args.model_path)

if __name__ == "__main__":
    try:
        trainer = PPOTrainer()
        episode = 0
        while trainer.args.episodes < 1 or trainer.args.episodes > episode:
            trainer.train()
            episode += 1
    except KeyboardInterrupt:
        pass 