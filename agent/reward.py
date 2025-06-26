from dataclasses import dataclass
from typing import Callable

@dataclass
class CalcRewardOpts:
    min_wall_distance: float
    velocity_scalar: float
    crashed: bool
    checkpoint_activated: bool

CalcRewardFunc = Callable[[CalcRewardOpts], float]

def calc_reward_stage_0(opts: CalcRewardOpts) -> float:
    # Encourage motion gently, light wall penalty
    reward = 0.0

    if opts.min_wall_distance < 20.0:
        reward -= (20.0 - opts.min_wall_distance) * 0.05

    reward += opts.velocity_scalar * 0.5

    if opts.checkpoint_activated:
        reward += 3.0

    return reward * 0.005

def calc_reward_stage_1(opts: CalcRewardOpts) -> float:
    # Balanced speed and wall awareness
    reward = 0.0

    reward += opts.velocity_scalar * 0.2

    if opts.min_wall_distance < 25.0:
        reward -= (25.0 - opts.min_wall_distance) * 0.1

    if opts.checkpoint_activated:
        reward += 2.0

    if opts.crashed:
        reward -= 10.0

    return reward  * 0.005


def calc_reward_stage_2(opts: CalcRewardOpts) -> float:
    reward = 0.0

    wall_safe_threshold = 20.0

    if opts.min_wall_distance < wall_safe_threshold:
        reward -= (wall_safe_threshold - opts.min_wall_distance) * 0.5

    if opts.min_wall_distance >= wall_safe_threshold:
        reward += opts.velocity_scalar * 4
    else:
        reward += opts.velocity_scalar * 1

    if opts.checkpoint_activated:
        reward += 15

    if opts.crashed:
        reward -= 50

    return reward  * 0.005

reward_strategies: list[CalcRewardFunc] = [
    calc_reward_stage_0,
    calc_reward_stage_1,
    # calc_reward_stage_2,
]