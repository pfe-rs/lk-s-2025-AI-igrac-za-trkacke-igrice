import dataclasses
import json
import math
import logging
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Any, SupportsFloat

import numpy as np
import gymnasium as gym
from pygame import Vector2

from agent.ppo.reward import DefaultRewarder, Rewarder
from agent.ppo.state import INPUTS_COUNT, RAYS_COUNT, EnvState
from agent.ppo.utils import (
    clamp, Segment, TupleVec, angle_diff, cut_segs, get_line_in_between,
    get_seg_angle, get_segs_intersection, orient_same_if_parallel,
    orient_toward_reference, select_similar_angle_seg, vec2_to_tuple
)
from agent.utils import LevelManager
from agent.const import *
from game.car import Car, car_from_parameters
from game.common import line_to_tuple
from game.ray import Ray, RayHit, calc_rays, calculate_ray_hits, min_opposite_ray_pair
from quad_func import get_chosen_ones


ObservationT = np.ndarray
ActionT = np.ndarray


logging.basicConfig(level=logging.INFO)


class Env(gym.Env[ObservationT, ActionT]):
    def __init__(
        self,
        car: Car,
        level_manager: LevelManager,
        rewarder: Rewarder,
        inputs_count: int = INPUTS_COUNT,
        max_steps: int = 16200,
        init_lives_count: int = 30,
        debug_dump_dir: Path | None = None,
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
    
        self.max_distance = math.hypot( # max distance for the level
            self.level.proportions[0],
            self.level.proportions[1],
        )

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

        # Navigation

        # basically serves as midlines count + next id
        # incremented on every new midline
        self.next_midline_id: int = 0 
        # segment:id so we don't process midlines twice
        self.midlines_cache: dict[tuple[TupleVec, TupleVec], int] = {}
        # id : processed midline in correct orientation
        self.midlines: dict[int, Segment] = {} # normalized
        # current direction by midline
        self.current_midline: Segment | None = None
        self.direction: float | None = None

        # state logs
        self.debug_dump_dir: Path | None = debug_dump_dir
    
    def __del__(self):
        if hasattr(self, '_debug_log_file') and not self._debug_log_file.closed:
            self._debug_log_file.close()

    def _setup_debug_log(self):
        if not self.debug_dump_dir:
            return

        self.debug_dump_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self, '_debug_log_file') and not self._debug_log_file.closed:
            self._debug_log_file.close()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._debug_log_path = self.debug_dump_dir / f'ppo_state_log_{timestamp}.json'
        self._debug_log_file = self._debug_log_path.open('w', encoding='utf-8')

    def _write_debug_log(self, data: dict):
        if not self.debug_dump_dir:
            return

        if hasattr(self, '_debug_log_file') and not self._debug_log_file.closed:
            json.dump(data, self._debug_log_file)
            self._debug_log_file.write('\n')

    def _calc_ray_hits(self) -> list[RayHit]:
        """Calculate normalized distances to walls for each ray"""
        rays = [
            Ray(ray.length, ray.origin + self.car.pos, ray.rotation + self.car.ori)
            for ray in self.rays
        ]
        
        ray_hits = calculate_ray_hits(rays, self.chosen_walls)
        return ray_hits
    
    def _upd_intersections(self, ray_hits: list[RayHit]):
        intersections = [1.0] * RAYS_COUNT
        for ray_hit in ray_hits:
            intersections[ray_hit.ray_index] = clamp(
                0, 1, ray_hit.distance / self.max_distance
            )
        self.state.intersections = intersections
    
    def _upd_car_state(self) -> None:
        """Update car-related state variables"""
        self.state.crashed = 0
        self.state.car_friction = clamp(0, 1, self.car.friction_coeff / max_car_ni)
        self.state.car_pull = clamp(0, 1, self.car.pull / max_car_pull)
        self.state.car_velocity_x = clamp(-1, 1, self.car.vel.x / self.max_distance)
        self.state.car_velocity_y = clamp(-1, 1, self.car.vel.y / self.max_distance)
        self.state.car_rotation_sin = math.sin(self.car.ori)
        self.state.car_rotation_cos = math.cos(self.car.ori)

    def _get_midline(self, ray_hits: list[RayHit]) -> Segment | None:
        self.min_opposite_hits = min_opposite_ray_pair(ray_hits, RAYS_COUNT)
        if not self.min_opposite_hits:
            return
        return get_line_in_between(
            line_to_tuple(self.min_opposite_hits[0].wall),
            line_to_tuple(self.min_opposite_hits[1].wall),
        )
    
    def _process_midline(self, midline: Segment) -> Segment | None:
        processed_midline: Segment | None = None
        midline_key = (vec2_to_tuple(midline[0]), vec2_to_tuple(midline[1]))
        if midline_key in self.midlines_cache:
            midline_id = self.midlines_cache[midline_key]
            processed_midline = self.midlines[midline_id]
            return processed_midline
        if self.next_midline_id > 0: # prev midline exists
            prev_midline = self.midlines[self.next_midline_id-1]
            intersection = get_segs_intersection(prev_midline, midline)
            if intersection:
                midline_candidate_0, midline_candidate_1 = cut_segs(midline, intersection)
                processed_midline = select_similar_angle_seg(prev_midline, midline_candidate_0, midline_candidate_1)
            else:
                parallel_midline = orient_same_if_parallel(prev_midline, midline, 2 * math.pi)
                if parallel_midline:
                    processed_midline = parallel_midline
                else: 
                    processed_midline = orient_toward_reference(prev_midline[1], midline)
        else:
            midline_angle = get_seg_angle(midline)

            init_rotation = self.car.ori
            if angle_diff(init_rotation, midline_angle) < angle_diff(init_rotation, midline_angle + math.pi):
                processed_midline = midline
            else:
                processed_midline = (midline[1], midline[0])
        # add new midline to cache
        self.midlines_cache[midline_key] = self.next_midline_id
        self.midlines[self.next_midline_id] = processed_midline
        self.next_midline_id += 1
        return processed_midline

    
    def _upd_state(self) -> None:
        self._upd_car_state()
        decided_quad = self.car.decide_quad(self.level)
        self.chosen_walls = get_chosen_ones(self.level.walls, self.level.boolean_walls, decided_quad)
        ray_hits = self._calc_ray_hits()
        self._upd_intersections(ray_hits)

        midline = self._get_midline(ray_hits)
        if not midline:
            # Fallback to car's forward direction if no guideline available
            self.state.direction_cos = math.cos(self.car.ori)
            self.state.direction_sin = math.sin(self.car.ori)
            return

        processed_midline = self._process_midline(midline)
        if processed_midline: 
            self.direction = get_seg_angle(processed_midline)
            self.state.direction_cos = math.cos(self.direction)
            self.state.direction_sin = math.sin(self.direction)

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

        self._upd_state()

        terminated: bool = False
        if self.car.intersects_line(self.chosen_walls):
            crashed = True
            self.state.crashed = 1
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

        debug_dict = dataclasses.asdict(self.state)
        debug_dict["reward"] = reward
        debug_dict["step"] = self.steps
        self._write_debug_log(debug_dict)

        observation = np.array(self.state.flatten(), dtype=np.float32)
        info = {
            "score": self.score,
            "steps": self.steps,
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
        self._setup_debug_log()
        self.level = self.level_manager.random()[0]
        self._on_level_update()

        self.steps = 0
        self.score = 0
        self.lives_count = self.init_lives_count
        self.run = True

        self._upd_state()

        if self.run_reward_history:
            avg = sum(self.run_reward_history) / len(self.run_reward_history)
            self.avg_reward_history.append(avg)
        self.run_reward_history = []

        return np.array(self.state.flatten(), dtype=np.float32), {}


def env_factory(levels_path: Path) -> Env:
    car_params = (5, 40, 20, [100, 200, 255], 4000, 10, (0, 0, 0), 5)
    car = car_from_parameters(car_params)
    return Env(car, LevelManager(levels_path), DefaultRewarder(), debug_dump_dir=Path("logs/"))
