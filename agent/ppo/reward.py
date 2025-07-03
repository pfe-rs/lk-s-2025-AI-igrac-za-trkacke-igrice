from abc import abstractmethod, ABC
from dataclasses import dataclass
import math

from agent.ppo.state import EnvState

class Rewarder(ABC):
    @abstractmethod
    def reset(self):
        pass
    @abstractmethod
    def calc_reward(self, state: EnvState) -> float:
        pass
   
@dataclass
class RewarderConfig:
    min_speed: float = 0.005
    max_speed: float = 0.25
    velocity_reward_scale: float = 0.1
    low_speed_penalty: float = -0.1
    wall_threshold: float = 0.015
    max_wall_penalty: float = -0.5 
    crash_penalty: float = -1
    speed_scale: float = 0.3

class DefaultRewarder(Rewarder):
    def __init__(self, cfg: RewarderConfig = RewarderConfig()):
        self.cfg = cfg

    def reset(self): pass

    def calc_reward(self, s: EnvState) -> float:
        if s.crashed:
            return self.cfg.crash_penalty

        r = 0.0
        r += self._speed_alignment_reward(s)
        # r += self._wall_proximity_penalty(s)
        return r - 0.1

    def _speed_alignment_reward(self, s: EnvState) -> float:
        vx, vy = s.car_velocity_x, s.car_velocity_y
        speed = math.hypot(vx, vy)

        if speed < self.cfg.min_speed:
            penalty = (1 - (speed / self.cfg.min_speed)) * self.cfg.low_speed_penalty
            return penalty

        # unit velocity vector
        ux: float = vx / speed
        uy: float = vy / speed
        # how well aligned with track direction
        dot = max(min(ux * s.direction_cos + uy * s.direction_sin, 1.0), -1.0)
        angle = math.acos(dot)  # [0, pi]
        dir_reward = 1 - (angle / math.pi) - 0.5 # [-0.5, 0.5]

        # Might be used later
        if dir_reward > 0:
            dir_reward += min(speed, self.cfg.max_speed) * self.cfg.speed_scale
        else:
            dir_reward -= min(speed, self.cfg.max_speed) * self.cfg.speed_scale
        return dir_reward

    def _wall_proximity_penalty(self, s: EnvState) -> float:
        if not s.intersections:
            return 0.0

        d = min(s.intersections)
        if d >= self.cfg.wall_threshold:
            return 0.0

        severity = 1 - (d / self.cfg.wall_threshold)
        return self.cfg.max_wall_penalty * severity