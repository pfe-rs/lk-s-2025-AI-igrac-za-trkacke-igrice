from collections import deque
import math
from pathlib import Path
import numpy as np
from typing import Any, SupportsFloat
import gymnasium as gym
from pygame import Vector2
from agent.ppo import guidelines
from agent.ppo.reward import DefaultRewarder, Rewarder
from game.car import Car, car_from_parameters
from agent.ppo.state import INPUTS_COUNT, RAYS_COUNT, EnvState
from agent.utils import LevelManager
from agent.const import *
from game.common import line_to_tuple
from game.ray import calc_rays, calculate_ray_hits, group_hits_by_opposite_pairs, min_opposite_ray_pair
from quad_func import get_chosen_ones
import logging

ObservationT = np.ndarray
ActionT = np.ndarray

logging.basicConfig(level=logging.INFO)


def clamp(min_v: float, max_v: float, val: float) -> float:
    return max(max_v, min(min_v, val))

def get_line_in_between(
    l1: tuple[Vector2, Vector2],
    l2: tuple[Vector2, Vector2],
) -> tuple[Vector2, Vector2]:
    return (
        (l1[0] + l2[0]) * 0.5,
        (l1[1] + l2[1]) * 0.5,
    )


class Env(gym.Env[ObservationT, ActionT]):
    def __init__(
        self,
        car: Car,
        level_manager: LevelManager,
        rewarder: Rewarder,
        inputs_count: int = INPUTS_COUNT,
        max_steps: int = 16200,
        init_lives_count: int = 30
    ) -> None:
        super(Env, self).__init__()
        self.action_space = gym.spaces.MultiBinary(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(inputs_count,),
            dtype=np.float32,
        )

        self.level_manager = level_manager
        self.level = level_manager.random()[0]
        self.FPS = self.level.FPS
        self.car = car

        self.level = self.level_manager.random()[0]
        self._on_level_update()

        self.state = EnvState(
            car_mass=clamp(0, 1, self.car.mass/max_car_mass),
            car_length=clamp(0, 1, self.car.length/max_car_length),
            car_width=clamp(0, 1, self.car.width/max_car_width),
            car_brake_friction_multiplier=clamp(0, 1, self.car.brake_friction_multiplier/max_car_k),
        )

        self.steps = 0
        self.score = 0
        self.max_steps: int = max_steps
        self.run = True

        self.rays = calc_rays(RAYS_COUNT,
            0, Vector2(0, 0), self.max_distance
        )

        self.rewarder = rewarder
        self.run_reward_history = [] # inside one run
        self.avg_reward_history = [] # between runs
        # Rewinding
        self.position_history = deque(maxlen=20)
        self.init_lives_count = init_lives_count
        self.lives_count = init_lives_count
        self.rewind_threshold = 200

        self.max_distance = math.hypot( # max distance for the level
            self.level.proportions[0],
            self.level.proportions[1],
        )
        # if car rotates too much away from prev ok direction -- we keep it
        self.prev_ok_direction = self.car.ori
        
        # Initialize guideline system with car and level parameters
        self.guideline = None
        self._build_guidelines()

    def _build_guidelines(self):
        """Build the global guideline system from wall middlelines."""
        # Generate sample rays to find middlelines across the track
        sample_rays = calc_rays(RAYS_COUNT, 0, self.car.pos, self.max_distance)
        for ray in sample_rays:
            ray.origin = self.car.pos
            ray.rotation = self.car.ori
        
        # Get initial ray hits to establish middlelines
        ray_hits = calculate_ray_hits(sample_rays, self.level.walls)
        opposite_hits_pairs = group_hits_by_opposite_pairs(ray_hits, RAYS_COUNT)
        
        # Extract middlelines from opposite wall pairs
        middlelines = []
        for hit_a, hit_b in opposite_hits_pairs:
            if hit_a and hit_b:
                middleline = get_line_in_between(line_to_tuple(hit_a.wall), line_to_tuple(hit_b.wall))
                middlelines.append(middleline)
        
        # Build consistent guideline path with dynamic parameters
        self.guideline = guidelines.GuidelineBuilder(
            self.car.pos, 
            self.car.ori,
            car_length=self.car.length,
            level_scale=self.max_distance
        )
        self.guideline.build(middlelines)
    
    def _upd_state(self) -> None:
        decided_quad = self.car.decide_quad(self.level)
        self.chosen_walls = get_chosen_ones(self.level.walls, self.level.boolean_walls, decided_quad)

        intersections: list[float] = RAYS_COUNT * [1.0] # normalized distances to walls
        for ray in self.rays:
            ray.origin = self.car.pos
            ray.rotation = self.car.ori

        ray_hits = calculate_ray_hits(self.rays, self.chosen_walls)
        
        for ray_hit in ray_hits: # append normalized distance to walls
            intersections[ray_hit.ray_index] = clamp(0, 1, ray_hit.distance/self.max_distance)
        self.state.intersections = intersections
        
        self.state.car_friction = clamp(0, 1, self.car.friction_coeff/max_car_ni)
        self.state.car_pull = clamp(0, 1, self.car.pull/max_car_pull)

        self.state.car_velocity_x = clamp(-1, 1, self.car.vel.x/self.max_distance)
        self.state.car_velocity_y = clamp(-1, 1, self.car.vel.y/self.max_distance)
        self.state.car_rotation_sin = clamp(-1, 1, math.sin(self.car.ori))
        self.state.car_rotation_cos = clamp(-1, 1, math.cos(self.car.ori))

        self.min_opposite_hits = min_opposite_ray_pair(ray_hits, RAYS_COUNT)
        if not self.min_opposite_hits:
            # Fallback to car's forward direction if no guideline available
            self.state.direction_cos = clamp(-1, 1, math.cos(self.car.ori))
            self.state.direction_sin = clamp(-1, 1, math.sin(self.car.ori))
        else:
            # TODO: Get direction
            pass


    def _on_level_update(self):
        self.max_distance = math.hypot( # max distance for the level
            self.level.proportions[0],
            self.level.proportions[1],
        )
        self.car.to_start(
            Vector2(self.level.location[0], self.level.location[1]),
            self.level.location[2]
        )

    def step(self, action: ActionT) -> tuple[ObservationT, SupportsFloat, bool, bool, dict[str, Any]]:
        # Process binary actions
        if action[0]: self.car.gas()
        if action[1]: self.car.brake()
        if action[2]: self.car.steer_left()
        if action[3]: self.car.steer_right()

        # Physics and state update
        self.car.normalize_orientation()
        self.car.apply_friction(self.level.g)
        self.car.accelerate(self.FPS)
        self.car.step(self.FPS)

        if self.steps % 5 == 0: # every fiftth frame
            self.position_history.append((self.car.pos, self.car.ori))

        decided_quad = self.car.decide_quad(self.level)
        self.chosen_walls = get_chosen_ones(self.level.walls,self.level.boolean_walls,decided_quad)

        self._upd_state()

        terminated: bool = False
        if self.car.intersects_line(self.chosen_walls):
            crashed = True
            # TODO: Rewind
            # if self.position_history and self.lives_count > 0 and self.steps >= self.rewind_threshold:
            #     self.steps_since_rewind = 0
            #     self.lives_count -= 1
            #     self.car.to_start(self.position_history[0], self.position_history[1])
            # else:
            self.run = False
            terminated = True

        reward = self.rewarder.calc_reward(self.state)
        self.run_reward_history.append(reward)

        observation = np.array(self.state.flatten(), dtype=np.float32)
        info = {
            "score": self.score,
            "steps": self.steps,
            "checkpoint": self.check_number,
        }
        if self.steps >= self.max_steps and self.max_steps > 0:
            truncated = True
            return observation, reward, False, truncated, info
        
        self.steps += 1
        # self.steps_since_rewind += 1
        return observation, reward, terminated, False, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObservationT, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.level = self.level_manager.random()[0]
        self._on_level_update()

        self.steps = 0
        self.score = 0
        self.lives_count = self.init_lives_count
        self.run = True
        self.check_number = 0

        # Rebuild guidelines for new level
        self._build_guidelines()

        self._upd_state()

        if self.run_reward_history:
            avg = sum(self.run_reward_history) / len(self.run_reward_history)
            self.avg_reward_history.append(avg)
        self.run_reward_history = []

        return np.array(self.state.flatten(), dtype=np.float32), {}


def env_factory(levels_path: Path) -> Env:
    car_params = (5, 40, 20, [100, 200, 255], 4000, 10, (0, 0, 0), 5)
    car = car_from_parameters(car_params)
    return Env(car, LevelManager(levels_path), DefaultRewarder())
