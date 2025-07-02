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
    min_rewarded_velocity = 0.005
    max_rewarded_velocity = 0.25
    velocity_reward_scale: float = 1
    angle_reward_scale: float = 1
    crash_penalty: float = -7

class DefaultRewarder(Rewarder):
    def __init__(self, config: RewarderConfig = RewarderConfig()):
        self.config = config or RewarderConfig()
        self.prev_forward_progress = 1.0
        
    def reset(self):
        self.prev_forward_progress = 1.0
        
    def calc_reward(self, state: EnvState) -> float:
        reward = 1.0
        speed = math.sqrt(state.car_velocity_x**2 + state.car_velocity_y**2)
        
        if speed > self.config.min_rewarded_velocity:
            # Calculate velocity direction vector
            vel_dir_x = state.car_velocity_x / speed
            vel_dir_y = state.car_velocity_y / speed
            
            # Calculate angle difference between velocity and track direction
            dot_product = (
                vel_dir_x * state.direction_cos +
                vel_dir_y * state.direction_sin
            )
            dot_product = max(min(dot_product, 1.0), -1.0)
            angle_diff = math.acos(dot_product)

            forward_velocity = (
                state.car_velocity_x * state.direction_cos +
                state.car_velocity_y * state.direction_sin
            )
            
            direction_reward = self.config.angle_reward_scale * (math.pi - abs(angle_diff))
            
            reward += direction_reward + forward_velocity * 2
        
        if min(state.intersections) < 0.01:
            reward += -0.5
            
        # --- Crash Penalty ---
        if state.crashed == 1:
            reward += self.config.crash_penalty
        
        # --- Stability: Clip reward to reasonable bounds ---
        return reward - 2
