from dataclasses import dataclass
import math
from typing import Callable

@dataclass
class CalcRewardOpts:
    steps_passed_since_rewind: float
    min_wall_distance: float # from center of the car
    velocity_scalar: float # normalized (0-1)
    crashed: bool
    checkpoint_activated: bool

CalcRewardFunc = Callable[[CalcRewardOpts], float]

def calc_reward_stage_0(opts: CalcRewardOpts) -> float:
    reward = 0.0

    # Wall distance penalty (less aggressive falloff)
    wall_penalty = max(0.0, 1.0 - (opts.min_wall_distance / 40.0))
    reward -= wall_penalty * opts.velocity_scalar * 0.2  # was 0.4

    # Encourage some velocity when not too close to walls
    if opts.min_wall_distance > 10.0:
        reward += opts.velocity_scalar * 0.3 

    # Small penalty for idling
    if opts.velocity_scalar < 0.1:
        reward -= 0.05

    # Time survived reward (soft bonus over time)
    reward += min(1.0, opts.steps_passed_since_rewind / 1000.0) * 0.2

    # Checkpoint reward scaled by velocity (to reward fast correct driving)
    if opts.checkpoint_activated:
        reward += 1.0 + opts.velocity_scalar  # 1.0–2.0

    # Crash penalty (shaped sigmoid, capped)
    if opts.crashed:
        t = opts.steps_passed_since_rewind / 500
        crash_penalty = 1.0 + 1.0 / (1.0 + math.exp(6 * (t - 0.5)))
        crash_penalty = min(crash_penalty, 2.0)
        reward -= crash_penalty

    return reward


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

    return reward  * 0.5


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

    return reward  * 0.5

reward_strategies: list[CalcRewardFunc] = [
    calc_reward_stage_0,
    # calc_reward_stage_1,
    # calc_reward_stage_2,
]
