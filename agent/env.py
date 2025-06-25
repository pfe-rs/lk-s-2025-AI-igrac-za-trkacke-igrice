from dataclasses import dataclass, field
import math
import numpy as np
from typing import Any, SupportsFloat
import gymnasium as gym
from ClassesML2 import Car, Level, new_ray_intersection
from agent.reward import CalcRewardFunc, CalcRewardOpts
from common.const import *
from quad_func import get_chosen_ones

ObservationT = np.ndarray
ActionT = np.ndarray

def clamp(val: float) -> float:
    return max(0.0, min(1.0, val))

@dataclass
class EnvState: # NOTE: States have to be normalized
    car_mass: float = 0.0
    car_length: float = 0.0
    car_width: float = 0.0
    car_friction: float = 0.0
    car_brake_friction_multiplier: float = 0.0
    car_pull: float = 0.0
    car_rotation_sin: float = 0.5
    car_rotation_cos: float = 0.5
    car_velocity_x: float = 0.5
    car_velocity_y: float = 0.5
    # normalized distances to walls
    intersections: list[float] = field(default_factory=list)
    # not-normalized distances to walls, used for calculating reward
    # do not use for training ai
    # not in the flatten output
    _intersections: list[float] = field(default_factory=list)

    def flatten(self) -> list[float]:
        return [
            self.car_mass,
            self.car_length,
            self.car_width,
            self.car_friction,
            self.car_brake_friction_multiplier,
            self.car_pull,
            self.car_rotation_sin,
            self.car_rotation_cos,
            self.car_velocity_x,
            self.car_velocity_y,
            *self.intersections,
        ]


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

        self.state = EnvState(
            car_mass=clamp(self.car.mass/max_car_mass),
            car_length=clamp(self.car.length/max_car_length),
            car_width=clamp(self.car.width/max_car_width),
            car_brake_friction_multiplier=(self.car.k/max_car_k),
        )
        self.score = 0
        self.steps = 0
        self.run = True

        self.current_checkpoint_idx = 0
    
    def _upd_state(self) -> None:
        intersections: list[float] = [] # not-normalized distances to the walls
        intersections.extend(new_ray_intersection(self.level.proportions[0],
                                        self.level.proportions[1], 
                                        self.car.x, 
                                        self.car.y,
                                        self.car.ori,
                                        self.ray_number,
                                        self.level.walls))
        intersections.extend(new_ray_intersection(self.level.proportions[0],
                                        self.level.proportions[1], 
                                        self.car.x, 
                                        self.car.y,
                                        math.atan2(self.car.vy, self.car.vx),
                                        self.ray_number,
                                        self.level.walls))
        # allocating normalized array
        intersections_normalized: list[float] = [0] * len(intersections)
        for i in range(len(intersections)):
            intersections_normalized[i]=clamp((intersections[i] / 2000 + 1) / 2)

        self.state.car_friction = clamp(self.car.ni/max_car_ni)
        self.state.car_pull = clamp(self.car.pull/max_car_pull)
        self.state.car_velocity_x = clamp(self.car.vx/max_car_vx)
        self.state.car_velocity_y = clamp(self.car.vy/max_car_vy)
        self.state.car_rotation_sin = clamp((math.sin(self.car.ori)+1)/2)
        self.state.car_rotation_cos = clamp((math.cos(self.car.ori)+1)/2)
        self.state.intersections = intersections_normalized
        self.state._intersections = intersections

    def _get_state(self) -> EnvState:
        self._upd_state()
        return self.state

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

        decided_quad = self.car.decide_quad(self.level)
        self.chosen_walls = get_chosen_ones(self.level.walls,self.level.boolean_walls,decided_quad)

        self._upd_state()
        state = self._get_state()

        min_wall_distance: float = min(state._intersections)
        velocity_scalar: float = math.hypot(self.car.vx, self.car.vy)

        crashed: bool = False
        if self.car.wallinter(self.chosen_walls):
            crashed = True
            self.run = False

        # check if checkpoint activated
        checkpoint_activated = False
        if self.car.checkinter([self.level.checkpoints[self.check_number]]):
            self.check_number+=1
            if self.check_number>=len(self.level.checkpoints):
                self.check_number-=len(self.level.checkpoints)
            checkpoint_activated = True
            self.score += 1

        self.steps += 1
        
        reward = self._calc_reward(CalcRewardOpts(
            min_wall_distance,
            velocity_scalar,
            crashed,
            checkpoint_activated,
        ))

        raise Exception("Not implemented")
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObservationT, dict[str, Any]]:
        return super().reset(seed=seed, options=options)
