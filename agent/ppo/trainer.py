from pathlib import Path
from torch import nn

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback, StopTrainingOnNoModelImprovement
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
        
        # Create environment with wrappers
        self.env = DummyVecEnv([lambda: env_factory(self.args.levels_path)])
        self.env = VecMonitor(self.env)
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        
        # Define network architecture
        # NOTE: sb3 automatically adds input, output layers and sigmoid on the exit
        # also initializes weights
        policy_kwargs = {
            "net_arch": {
                "pi": [128,128, 64, 64],
                "vf": [128,128, 64, 64]
            },
            "activation_fn": nn.LeakyReLU,
            # applies orthogonal initialization to the linear layers
            "ortho_init": True,
        }
        
        model_path = self.args.model_path
        last_checkpoint_path = get_latest_checkpoint(self.args.checkpoints_path)
        load_path: Path | None = None
        if last_checkpoint_path:
            load_path = last_checkpoint_path
        elif model_path.exists():
            load_path = model_path

        if load_path:
            print(f"Loading model {last_checkpoint_path}")
            self.model = PPO.load(
                load_path, 
                self.env, 
                policy_kwargs=policy_kwargs,
                n_steps=2048,
                batch_size=64,
                n_epochs=self.args.epochs,
                verbose=1,
                device="cpu",
            )
        else:
            print(f"Training new model at {model_path}")
            self.model = PPO(
                "MlpPolicy",
                self.env,
                policy_kwargs=policy_kwargs,
                ent_coef=0.01, # Entropy coefficient for exploration. Default 0
                vf_coef=4e-3, # increased since value function wasn't learning
                learning_rate=1e-3,
                n_steps=2048,
                batch_size=64,
                n_epochs=self.args.epochs,
                verbose=1,
                device="cpu",
            )
            
        # Set common parameters for both new and loaded models
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
            n_eval_episodes=10,
            verbose=1,
            deterministic=True,
            callback_after_eval=StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=15,
                min_evals=2,
                verbose=1
            )
        )

        self.callbacks = CallbackList([
            checkpoint_cb,
            eval_cb,
        ])
    
    def train(self):
        try:
            self.model.learn(
                total_timesteps=self.args.total_timesteps,
                reset_num_timesteps=False,
                callback=self.callbacks,
                tb_log_name="PPO",
            )
        finally:
            print("Saving final model")
            self.model.save(self.args.model_path)

if __name__ == "__main__":
    trainer = PPOTrainer()
    try:
        trainer.train()
    except KeyboardInterrupt:
        pass