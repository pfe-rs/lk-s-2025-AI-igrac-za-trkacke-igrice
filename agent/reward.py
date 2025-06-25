from dataclasses import dataclass
from typing import Callable

@dataclass
class CalcRewardOpts:
    min_wall_distance: float
    velocity_scalar: float
    crashed: bool
    checkpoint_activated: bool

CalcRewardFunc = Callable[[CalcRewardOpts], float]

def calc_reward_stage_0(opts: CalcRewardFunc) -> float:
    r"""
    Just train it to go, reward for velocity is bigger, don't care if crashed,
    additional reward if checkpoint activated. Don't care about distance to the walls
    to make it not "afraid" to move
    """
    return opts.velocity_scalar * 2 + (10 if opts.checkpoint_activated else 0)

def calc_reward_stage_1(opts: CalcRewardFunc) -> float:
    r"""
    Train it to don't hit walls, to collect checkpoints.
    Bigger negative rewards for crashing. Velocity is still rewarded, but smaller
    """
    reward = opts.velocity_scalar
    if opts.crashed:
        reward -= 20
    if opts.min_wall_distance < 5:
        reward -= (5 - opts.min_wall_distance)  # penalize closeness
    if opts.checkpoint_activated:
        reward += 10
    return reward

def calc_reward_stage_2(opts: CalcRewardFunc) -> float:
    r"""
    Reward for going as fast as possible. No reward for slow speeds
    Additional small reward for checkpoints
    """
    return 0