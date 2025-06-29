from dataclasses import dataclass
from pygame import Vector2 

THRESHOLD = 0

@dataclass
class GuidelinesCarData:
    position: Vector2
    rotation: float


class Guidelines:
    guidelines_points: list[Vector2] = []

    def __init__(self, car: GuidelinesCarData, walls: list[Vector2]) -> None:
        pass

    def get_direction(self, point: Vector2) -> Vector2:
        raise Exception("Not implemented")
