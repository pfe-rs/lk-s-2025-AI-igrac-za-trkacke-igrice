from abc import abstractmethod, ABC
import math

from agent.ppo.state import EnvState


class Rewarder(ABC):
    @abstractmethod
    def calc_reward(self, state: EnvState) -> float: pass


class DefaultRewarder(Rewarder):
    def __init__(self):
        self.stagnant_steps = 0

    def calc_reward(self, state: EnvState) -> float:
        vel_x, vel_y = state.car_velocity_x, state.car_velocity_y
        speed = math.sqrt(vel_x ** 2 + vel_y ** 2)

        dir_x, dir_y = state.direction_cos, state.direction_sin
        alignment = vel_x * dir_x + vel_y * dir_y
        projected_progress = max(0.0, alignment)

        min_dist = min(state.intersections)
        reward = 0.0

        # 1. Progress reward
        reward += projected_progress * 1.5

        # 2. Crash penalty
        if state.crashed >= 1.0:
            reward -= 2.0 * (1.0 + speed)
            return reward  # ends episode

        # 3. Wall penalty
        if min_dist < 0.2:
            reward -= 0.6 * (1.0 - min_dist)

        # 4. Directional alignment
        reward += alignment * 0.15

        # 5. Speed bonus
        if alignment > 0.85:
            zone_factor = min(min_dist / 0.6, 1.0)
            reward += speed * 0.3 * zone_factor

        # 6. Stagnation penalty
        if speed < 0.1:
            self.stagnant_steps += 1
            reward += -0.1 - 0.05 * self.stagnant_steps
            if self.stagnant_steps > 100:
                reward += -5.0  # force exploration
        else:
            self.stagnant_steps = 0

        # 7. Time penalty
        reward -= 0.005

        return reward
