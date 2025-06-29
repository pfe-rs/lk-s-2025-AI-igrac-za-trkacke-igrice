import math
from dataclasses import dataclass
from typing import Optional
from pygame import Vector2
import pygame
from game.car import Car
from game.common import Line

@dataclass
class Ray:
    length: float
    origin: Vector2
    rotation: float

@dataclass
class RayHit:
    ray_index: int
    intersection_point: Vector2
    distance: float
    wall: Line


def line_ray_intersection(ray: Ray, line: tuple[pygame.Vector2, pygame.Vector2]) -> Optional[Vector2]:
    # Get ray direction from rotation (assuming rotation is in radians)
    ray_dir = pygame.Vector2(math.cos(ray.rotation), math.sin(ray.rotation))
    ray_end = ray.origin + ray_dir * ray.length
    
    p1, p2 = line
    
    # Calculate direction vectors
    ray_vec = ray_end - ray.origin
    line_vec = p2 - p1
    
    # Calculate determinant (cross product in 2D using pygame's cross method)
    det = ray_vec.cross(line_vec)
    # Lines are parallel if determinant is 0
    if abs(det) < 1e-10:
        return None
    
    # Vector from ray origin to line start
    diff = p1 - ray.origin
    
    # Calculate parameters
    t = diff.cross(line_vec) / det  # Parameter for ray
    u = diff.cross(ray_vec) / det   # Parameter for line
    
    # Check if intersection is within ray bounds (0 <= t <= 1)
    # and within line segment bounds (0 <= u <= 1)
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection = ray.origin + ray_vec * t
        return intersection
    
    return None


def calc_rays(ray_count: int, base_rotation: float, origin: pygame.Vector2, ray_length: float, 
                 from_rotation: Optional[float] = None, to_rotation: Optional[float] = None) -> list[Ray]:
    rays = []
    if ray_count <= 0:
        return rays
    
    if ray_count == 1:
        if from_rotation is not None and to_rotation is not None:
            rays.append(Ray(ray_length, origin, from_rotation))
        else:
            rays.append(Ray(ray_length, origin, base_rotation))
    elif from_rotation is not None and to_rotation is not None:
        # Bounded range: distribute rays from from_rotation to to_rotation
        angular_range = to_rotation - from_rotation
        angular_spacing = angular_range / (ray_count - 1) if ray_count > 1 else 0
        
        for i in range(ray_count):
            ray_rotation = from_rotation + (i * angular_spacing)
            rays.append(Ray(ray_length, origin, ray_rotation))
    else:
        # Unbounded: distribute rays in full circle around base_rotation
        angular_spacing = (2 * math.pi) / ray_count
        
        for i in range(ray_count):
            ray_rotation = base_rotation + (i * angular_spacing)
            rays.append(Ray(ray_length, origin, ray_rotation))

    return rays


def calculate_ray_hits(rays: list[Ray], wall_lines: list[Line]) -> list[RayHit]:
    hits: list[RayHit] = []
    
    for ray_index, ray in enumerate(rays):
        closest_hit = None
        closest_distance = float('inf')
        closest_wall = None
        
        for wall in wall_lines:
            intersection = line_ray_intersection(ray,
                (Vector2(wall.x1, wall.y1), Vector2(wall.x2, wall.y2)),
            )
            if intersection:
                distance = ray.origin.distance_to(intersection)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_hit = intersection
                    closest_wall = wall
        
        if closest_hit is not None and closest_wall is not None:
            hits.append(RayHit(
                ray_index=ray_index,
                intersection_point=closest_hit,
                distance=closest_distance,
                wall=closest_wall
            ))
    
    return hits


def calculate_all_ray_hits(rays: list[Ray], wall_lines: list[Line]) -> list[list[RayHit]]:
    all_hits: list[list[RayHit]] = []
    
    for ray_index, ray in enumerate(rays):
        ray_hits = []
        
        for wall in wall_lines:
            intersection = line_ray_intersection(ray,
                (Vector2(wall.x1, wall.y1), Vector2(wall.x2, wall.y2)),
            )
            if intersection:
                distance = ray.origin.distance_to(intersection)
                ray_hits.append(RayHit(
                    ray_index=ray_index,
                    intersection_point=intersection,
                    distance=distance,
                    wall=wall
                ))
        
        # Sort hits by distance for this ray
        ray_hits.sort(key=lambda hit: hit.distance)
        all_hits.append(ray_hits)
    
    return all_hits


def get_opposite_ray_index(ray_index: int, total_rays: int) -> int:
    if total_rays % 2 != 0:
        raise ValueError("Opposite rays only defined for even number of rays")
    
    return (ray_index + total_rays // 2) % total_rays


def group_hits_by_opposite_pairs(hits: list[RayHit], total_rays: int) -> list[tuple[Optional[RayHit], Optional[RayHit]]]:
    if total_rays % 2 != 0:
        raise ValueError("Opposite ray pairing only works with even number of rays")
    
    hit_dict = {hit.ray_index: hit for hit in hits}
    
    pairs = []
    processed = set()
    
    for i in range(total_rays):
        if i in processed:
            continue
            
        opposite_i = get_opposite_ray_index(i, total_rays)
        
        ray_hit = hit_dict.get(i)
        opposite_hit = hit_dict.get(opposite_i)
        
        pairs.append((ray_hit, opposite_hit))
        processed.add(i)
        processed.add(opposite_i)
    
    return pairs

def min_opposite_ray_pair(hits: list[RayHit], total_rays: int) -> tuple[RayHit, RayHit] | None:
    opposite_pairs = group_hits_by_opposite_pairs(hits, total_rays)
    min_opposite_pairs_len = -1
    min_opposite_pairs_idx = -1
    for i in range(len(opposite_pairs)):
        opposite_pair = opposite_pairs[i]
        hit_a, hit_b = opposite_pair
        if not hit_a or not hit_b:
            continue
        len_sum = hit_a.distance + hit_b.distance
        if len_sum < min_opposite_pairs_len or min_opposite_pairs_len == -1:
            min_opposite_pairs_len = len_sum
            min_opposite_pairs_idx = i

    if min_opposite_pairs_idx == -1:
        return None
    hit_a, hit_b = opposite_pairs[min_opposite_pairs_idx]
    if not hit_a or not hit_b:
        raise ValueError("impossible condition")
    return (hit_a, hit_b)