from dataclasses import dataclass
from typing import Callable

@dataclass
class CalcRewardOpts:
    min_wall_distance: float
    velocity_scalar: float
    crashed: bool

type CalcRewardFunc = Callable[[CalcRewardOpts], float]

def calc_reward_stage_1():
    pass