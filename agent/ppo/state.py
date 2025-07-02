from dataclasses import dataclass, field
import dataclasses
import json

RAYS_COUNT = 16
CAR_STATIC_PARAMS_COUNT = 4
CAR_STATE_PARAMS_COUNT = 6
CAR_NAVIGATION_PARAMS_COUNT = 3 + RAYS_COUNT
INPUTS_COUNT = CAR_STATIC_PARAMS_COUNT + CAR_STATE_PARAMS_COUNT + CAR_NAVIGATION_PARAMS_COUNT

@dataclass
class EnvState:
    # Static params [0, 1]
    car_mass: float
    car_length: float
    car_width: float
    car_brake_friction_multiplier: float
    # Dynamic params [0, 1]
    car_pull: float = 0.5
    car_friction: float = 0.5 
    # [-1, 1]
    car_velocity_x: float = 0.0
    car_velocity_y: float = 0.0
    car_rotation_sin: float = 0.0
    car_rotation_cos: float = 0.0

    # Navigation params
    # direction car should go [-1, 1]
    direction_sin: float = 0.0
    direction_cos: float = 0.0
    # wall kissed [0 or 1]
    crashed: float = 0
    # distances to walls # [0, 1]
    intersections: list[float] = field(default_factory=list)

    def json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    def flatten(self) -> list[float]: # used as input for nn
        assert len(self.intersections) == RAYS_COUNT
        return [
            self.car_mass,
            self.car_length,
            self.car_width,
            self.car_brake_friction_multiplier,

            self.car_pull,
            self.car_friction,
            self.car_velocity_x,
            self.car_velocity_y,
            self.car_rotation_sin,
            self.car_rotation_cos,

            self.direction_sin,
            self.direction_cos,
            self.crashed,
            *self.intersections,
        ]

