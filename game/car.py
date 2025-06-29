import math
import pygame
from pygame import Color, Surface
from pygame.math import Vector2
from typing import List, Tuple, Union

from game.common import Line

class Car:
    def __init__(
        self,
        mass: float,
        length: float,
        width: float,
        color: Union[Tuple[int, int, int], Color],
        pull: float,
        base_friction_coeff: float = 5,
        location: Tuple[float, float, float] = (100, 100, 0.5 * math.pi),
        brake_friction_multiplier: float = 5
    ) -> None:
        self.mass = mass
        self.length = length
        self.width = width
        self.color = color

        self.pull = pull
        self.base_friction_coeff = base_friction_coeff
        self.brake_friction_multiplier = brake_friction_multiplier
        self.friction_coeff = base_friction_coeff
        self.is_braking = False  # Track braking state

        self.ori = location[2]
        self.pos = Vector2(location[0], location[1])
        self.vel = Vector2()
        self.acc = Vector2()

        self.forces: List[Vector2] = []
        
        # Cache for collision detection to avoid recreating Line objects
        self._car_lines_cache: List[Line] = [Line(0, 0, 1, 0) for _ in range(4)]

    def normalize_orientation(self) -> None:
        self.ori %= (2 * math.pi)

    def to_start(self, pos: Vector2, ori: float) -> None:
        self.ori = ori
        self.pos = pos
        self.vel = Vector2()
        self.acc = Vector2()
        self.forces.clear()
        self.is_braking = False
        self.friction_coeff = self.base_friction_coeff

    def get_rotated_corners(self) -> List[Vector2]:
        cos_ori = math.cos(self.ori)
        sin_ori = math.sin(self.ori)
        half_l, half_w = self.length / 2, self.width / 2
        local_corners = [
            (-half_l, -half_w),  # rear-left
            (half_l, -half_w),   # front-left
            (half_l, half_w),    # front-right
            (-half_l, half_w)    # rear-right
        ]
        rotated_corners = [
            Vector2(
                self.pos.x + x * cos_ori - y * sin_ori,
                self.pos.y + x * sin_ori + y * cos_ori,
            )
            for x, y in local_corners
        ]
        return rotated_corners

    def decide_quad(self, level) -> List[bool]:
        quads = [False] * 4
        w, h = level.proportions
        for corner in self.get_rotated_corners():
            if corner.x < w / 2 and corner.y < h / 2:
                quads[0] = True
            elif corner.x >= w / 2 and corner.y < h / 2:
                quads[1] = True
            elif corner.x < w / 2 and corner.y >= h / 2:
                quads[2] = True
            elif corner.x >= w / 2 and corner.y >= h / 2:
                quads[3] = True
        return quads

    def draw(self, screen: Surface) -> None:
        cos_ori = math.cos(self.ori)
        sin_ori = math.sin(self.ori)
        half_l, half_w = self.length / 2, self.width / 2
        local_corners = [
            (-half_l, -half_w),  # rear-left
            (half_l, -half_w),   # front-left
            (half_l, half_w),    # front-right
            (-half_l, half_w)    # rear-right
        ]
        rotated_corners = [
            Vector2(
                self.pos.x + x * cos_ori - y * sin_ori,
                self.pos.y + x * sin_ori + y * cos_ori,
            )
            for x, y in local_corners
        ]

        triangle_side = self.width
        triangle_height = (math.sqrt(3) / 2) * triangle_side

        local_triangle = [
            Vector2(triangle_height, 0),                  # Tip
            Vector2(0, -triangle_side / 2),               # Back left
            Vector2(0, triangle_side / 2),                # Back right
        ]

        rotated_triangle = [
            Vector2(
                self.pos.x + p.x * cos_ori - p.y * sin_ori,
                self.pos.y + p.x * sin_ori + p.y * cos_ori,
            )
            for p in local_triangle
        ]

        pygame.draw.polygon(screen, self.color, rotated_corners)
        pygame.draw.polygon(screen, (255, 0, 0), rotated_triangle)

    def update_physics(self, FPS: float, g: float) -> None:
        """Unified physics update method that handles all physics in correct order"""
        # Apply friction
        self.apply_friction(g)
        
        # Calculate acceleration from all forces
        total_force = sum(self.forces, start=Vector2())
        self.acc = (total_force / self.mass) / FPS
        
        # Update velocity and position
        self.vel += self.acc
        self.pos += self.vel / FPS
        
        # Clear forces for next frame
        self.forces.clear()
        
        # Reset friction coefficient if not braking
        if not self.is_braking:
            self.friction_coeff = self.base_friction_coeff

    def gas(self) -> None:
        direction = Vector2(math.cos(self.ori), math.sin(self.ori))
        self.forces.append(direction * self.pull)

    def apply_friction(self, g: float) -> None:
        speed = self.vel.length()
        if speed > 0:
            friction_mag = self.mass * g * self.friction_coeff
            friction = -self.vel.normalize() * friction_mag
            self.forces.append(friction)

    def brake(self) -> None:
        self.is_braking = True
        self.friction_coeff = self.base_friction_coeff * self.brake_friction_multiplier

    def release_brake(self) -> None:
        """Call this when brake is released"""
        self.is_braking = False

    def steer_left(self) -> None:
        self.ori -= 0.05

    def steer_right(self) -> None:
        self.ori += 0.05

    def max_speed(self, g: float) -> float:
        return self.pull / (self.base_friction_coeff * self.mass * g)

    def max_friction(self, g: float) -> float:
        return self.mass * g * self.base_friction_coeff
    
    def intersects_line(self, lines: list[Line]) -> bool:
        """Efficient collision detection using cached Line objects"""
        corners = self.get_rotated_corners()
        
        # Update cached line objects instead of creating new ones
        for i in range(4):
            corner1 = corners[i]
            corner2 = corners[(i + 1) % 4]
            
            # Update the existing Line object
            self._car_lines_cache[i].x1 = corner1[0]
            self._car_lines_cache[i].y1 = corner1[1]
            self._car_lines_cache[i].x2 = corner2[0]
            self._car_lines_cache[i].y2 = corner2[1]
        
        # Check intersections
        for car_line in self._car_lines_cache:
            for line in lines:
                if car_line.is_intersecting(line):
                    return True
        return False

    # Deprecated methods - kept for backward compatibility but recommend using update_physics instead
    def accelerate(self, FPS: float) -> None:
        """Deprecated: Use update_physics() instead"""
        total_force = sum(self.forces, start=Vector2())
        self.acc = (total_force / self.mass) / FPS
        self.vel += self.acc
        self.forces.clear()

    def step(self, FPS: float) -> None:
        """Deprecated: Use update_physics() instead"""
        self.pos += self.vel / FPS


def car_from_parameters(parameters):
    # Car(5,40,20,([100,200,255]),1000,10,level.location,3)
   return Car(
       mass=parameters[0],
       length=parameters[1], 
       width=parameters[2],
       color=parameters[3],
       pull=parameters[4],
       base_friction_coeff=parameters[5],
       location=parameters[6],
       brake_friction_multiplier=parameters[7]
   )