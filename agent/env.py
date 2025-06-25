from dataclasses import dataclass
import math
import numpy as np
from typing import Any, SupportsFloat
import gymnasium as gym
from ClassesML2 import Car, Level, new_ray_intersection
from agent.reward import CalcRewardFunc

ObservationT = np.ndarray
ActionT = np.ndarray

@dataclass
class EnvState:
    car_mass: float = 0
    car_length: int = 0 
    car_width: int = 0
    car_friction: float = 0
    car_brake_friction_multiplier: float = 0
    car_pull: float = 0
    car_rotation_sin: float = 1
    car_rotation_cos: float = 0
    car_velocity_x: float = 0
    car_velocity_y: float = 0
    intersections: list = []

    def flatten(self) -> list:
        norm = lambda val, div: max(0, min(1, val / div))
        norm_signed = lambda val, div: max(0, min(1, (val / div + 1) / 2))

        state = [
            norm(self.car_mass, 100),
            norm(self.car_length, 200),
            norm(self.car_width, 100),
            norm(self.car_friction, 200),
            norm(self.car_brake_friction_multiplier, 120),
            norm(self.car_pull, 10000),
            norm((self.car_rotation_sin + 1), 2),
            norm((self.car_rotation_cos + 1), 2),
            norm_signed(self.car_velocity_x, 1000),
            norm_signed(self.car_velocity_y, 1000),
        ]
        state.extend([norm_signed(i, 2000) for i in self.intersections])
        return state


class Env(gym.Env[ObservationT, ActionT]):
    def __init__(
        self,
        level: Level,
        car: Car,
        calc_reward: CalcRewardFunc,
        inputs_count: int = 24,
        rays_count: int = 7,
    ) -> None:
        super(Env, self).__init__()
        self.action_space = gym.spaces.MultiBinary(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(inputs_count,),
            dtype=np.float32,
        )

        self.level = level
        self.car = car
        self._calc_reward = calc_reward
        self.FPS = level.FPS
        self.ray_number=rays_count

        self.state = None
        self.score = 0
        self.steps = 0
        self.run = True

        self.current_checkpoint_idx = 0

    def step(self, action: ActionT) -> tuple[ObservationT, SupportsFloat, bool, bool, dict[str, Any]]:
        super().step(action)
        self.car.rfx = 0
        self.car.rfy = 0
        self.car.ni = self.car.bni

        # Process binary actions
        if action[0]: self.car.gas()
        if action[1]: self.car.brake()
        if action[2]: self.car.steerleft()
        if action[3]: self.car.steerright()

        # Physics and state update
        self.car.Ori()
        self.car.friction(self.level.g)
        self.car.ac(self.FPS)
        self.car.step(self.FPS)
        raise Exception("Not implemented")

    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObservationT, dict[str, Any]]:
        return super().reset(seed=seed, options=options)
