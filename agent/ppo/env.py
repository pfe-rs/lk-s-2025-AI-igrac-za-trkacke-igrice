from dataclasses import dataclass, field
import statistics
import math
from pathlib import Path
import numpy as np
from typing import Any, SupportsFloat
import gymnasium as gym
from ClassesML2 import Car, Level, new_ray_intersection
from Functions import car_from_parameters, level_loader
from agent.reward import CalcRewardFunc, CalcRewardOpts, reward_strategies
from common.const import *
from quad_func import get_chosen_ones
import logging

ObservationT = np.ndarray
ActionT = np.ndarray


logging.basicConfig(level=logging.INFO)

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
        reward_strategies: list[CalcRewardFunc] = reward_strategies,
        inputs_count: int = inputs_count,
        rays_count: int = rays_count,
        max_steps: int = 0,
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
        self.FPS = level.FPS
        self.ray_number=rays_count

        self.state = EnvState(
            car_mass=clamp(self.car.mass/max_car_mass),
            car_length=clamp(self.car.length/max_car_length),
            car_width=clamp(self.car.width/max_car_width),
            car_brake_friction_multiplier=(self.car.k/max_car_k),
        )
        self.current_checkpoint_idx = 0

        self.steps = 0
        self.max_steps: int = max_steps
        self.score = 0
        self.run = True

        # For choosing strategy
        self.reward_strategies = reward_strategies
        self.current_reward_strategy = 0
        # When this number is too big -- we change strategy
        self.min_runs_without_strategy_switch = 300
        self.small_change_streak = 0
        self.run_reward_history = [] # inside one run
        self.avg_reward_history = [] # between runs
    
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

        reward: float = self.reward_strategies[self.current_reward_strategy](CalcRewardOpts(
            min_wall_distance,
            velocity_scalar,
            crashed,
            checkpoint_activated,
        ))
        self.run_reward_history.append(reward)

        # logging.info(f"[Step {self.steps}] Reward Stage {self.current_reward_strategy} | "
        #          f"Reward: {reward:.5f} | "
        #          f"Velocity: {velocity_scalar:.2f} | "
        #          f"MinWallDist: {min_wall_distance:.2f} | "
        #          f"Crashed: {crashed} | "
        #          f"Checkpoint: {checkpoint_activated}")

        observation = np.array(state.flatten(), dtype=np.float32)
        info = {
            "score": self.score,
            "steps": self.steps,
            "checkpoint": self.check_number,
            "reward_strategy": self.current_reward_strategy
        }
        if self.steps >= self.max_steps and self.max_steps > 0:
            truncated = True
            return observation, reward, False, truncated, info
        
        self.steps += 1
        return observation, reward, crashed, False, info

    def _should_switch_reward_strategy(self) -> bool:
        if len(self.reward_strategies) <= 1:
            return False

        window = self.min_runs_without_strategy_switch
        if len(self.avg_reward_history) < 3 * window:
            return False

        recent = self.avg_reward_history[-window:]
        mid = self.avg_reward_history[-2 * window:-window]
        old = self.avg_reward_history[-3 * window:-2 * window]

        mean_recent = statistics.mean(recent)
        mean_mid = statistics.mean(mid)
        mean_old = statistics.mean(old)

        delta_1 = mean_mid - mean_old
        delta_2 = mean_recent - mean_mid

        small_change = abs(delta_2) < 0.001 * abs(mean_mid)
        negative_trend = delta_2 < -0.001 * abs(mean_mid)

        try:
            std_recent = statistics.stdev(recent)
        except statistics.StatisticsError:
            std_recent = 0.0
        flat_curve = std_recent < 0.01 * abs(mean_recent)

        return negative_trend or (small_change and flat_curve)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObservationT, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.car.tostart(self.level.location)
        self.steps = 0
        self.score = 0
        self.run = True
        self.check_number = 0

        self._upd_state()

        if self.run_reward_history:
            avg = sum(self.run_reward_history) / len(self.run_reward_history)
            self.avg_reward_history.append(avg)
        self.run_reward_history = []

        if self._should_switch_reward_strategy():
            self.current_reward_strategy = (self.current_reward_strategy + 1) % len(self.reward_strategies)

        return np.array(self.state.flatten(), dtype=np.float32), {"strategy": self.current_reward_strategy}


def env_factory(level_path: Path) -> Env:
    level = level_loader(level_path)
    car_params = (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)
    car = car_from_parameters(car_params)
    car.x = level.location[0]
    car.y = level.location[1] 
    return Env(level, car, reward_strategies=reward_strategies)
