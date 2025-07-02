from typing import Optional
from pygame import Vector2
import math

TupleVec = tuple[float, float]
Segment = tuple[Vector2, Vector2]

def vec2_to_tuple(vec: Vector2) -> TupleVec:
    return (vec.x, vec.y)

def clamp(min_v: float, max_v: float, val: float) -> float:
    return max(min_v, min(max_v, val))

def check_segments_eq(s1: Segment | None, s2: Segment | None) -> bool:
    if s1 is None or s2 is None:
        return s1 is s2
    return (s1[0] == s2[0] and s1[1] == s2[1]) or (s1[0] == s2[1] and s1[1] == s2[0])

def get_line_in_between(
    l1: Segment,
    l2: Segment,
) -> Segment:
    """Compute geometrically averaged line between two lines."""
    midpoint1 = (l1[0] + l1[1]) * 0.5
    midpoint2 = (l2[0] + l2[1]) * 0.5
    center = (midpoint1 + midpoint2) * 0.5

    dir1 = l1[1] - l1[0]
    dir2 = l2[1] - l2[0]
    avg_dir = dir1 + dir2
    if avg_dir.length() == 0:
        avg_dir = Vector2(1, 0)  # fallback

    avg_dir = avg_dir.normalize()
    avg_len = (dir1.length() + dir2.length()) * 0.25  # half-length
    offset = avg_dir * avg_len
    return (center - offset, center + offset)

def get_segs_intersection(s1: Segment, s2: Segment) -> Optional[Vector2]:
    def ccw(A, B, C):
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)
    
    A, B = s1
    C, D = s2

    if ccw(A, B, C) != ccw(A, B, D) and ccw(C, D, A) != ccw(C, D, B):
        denom = (B.x - A.x) * (D.y - C.y) - (B.y - A.y) * (D.x - C.x)
        if math.isclose(denom, 0.0):
            return None
        
        num_t = (A.y - C.y) * (D.x - C.x) - (A.x - C.x) * (D.y - C.y)
        t = num_t / denom
        return Vector2(A.x + t * (B.x - A.x), A.y + t * (B.y - A.y))
    
    return None


def get_seg_angle(seg: Segment) -> float:
    start, end = seg
    delta = end - start
    return math.atan2(delta.y, delta.x)

# returns two segments where first point is the provided point
# and second point is end/start of the original segment
def cut_segs(seg: Segment, point: Vector2) -> tuple[Segment, Segment]:
    return ((point, seg[0]), (point, seg[1]))

def angle_diff(a: float, b: float):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

def select_similar_angle_seg(orig_seg: Segment, seg1: Segment, seg2: Segment) -> Segment:
    orig_angle = get_seg_angle(orig_seg)
    angle1 = get_seg_angle(seg1)
    angle2 = get_seg_angle(seg2)

    diff1 = angle_diff(orig_angle, angle1)
    diff2 = angle_diff(orig_angle, angle2)

    return seg1 if diff1 < diff2 else seg2

def orient_toward_reference(ref: Vector2, seg: Segment) -> Segment:
    return seg if ref.distance_to(seg[0]) < ref.distance_to(seg[1]) else (seg[1], seg[0])

def angle_between(v1: Vector2, v2: Vector2) -> float:
    dot = max(min(v1.normalize().dot(v2.normalize()), 1.0), -1.0)
    return math.acos(dot)

def is_parallel(seg1: Segment, seg2: Segment, threshold: float) -> bool:
    dir1 = seg1[1] - seg1[0]
    dir2 = seg2[1] - seg2[0]

    if dir1.length() == 0 or dir2.length() == 0:
        return False  # can't define direction for zero-length segment
    angle = angle_between(dir1, dir2)
    return angle < threshold or abs(math.pi - angle) < threshold

def orient_same_if_parallel(seg1: Segment, seg2: Segment, threshold: float) -> Segment | None:
    dir1 = seg1[1] - seg1[0]
    dir2 = seg2[1] - seg2[0]

    if dir1.length() == 0 or dir2.length() == 0:
        return None
    angle = angle_between(dir1, dir2)
    if angle < threshold:
        return seg2
    elif abs(math.pi - angle) < threshold:
        return (seg2[1], seg2[0])
    else:
        return None