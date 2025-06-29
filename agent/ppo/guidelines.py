import math
from pygame import Vector2


# def get_line_in_between(
#     l1: tuple[Vector2, Vector2],
#     l2: tuple[Vector2, Vector2],
# ) -> tuple[Vector2, Vector2]:
#     """Compute geometrically averaged line between two lines."""
#     midpoint1 = (l1[0] + l1[1]) * 0.5
#     midpoint2 = (l2[0] + l2[1]) * 0.5
#     center = (midpoint1 + midpoint2) * 0.5

#     dir1 = l1[1] - l1[0]
#     dir2 = l2[1] - l2[0]
#     avg_dir = dir1 + dir2
#     if avg_dir.length() == 0:
#         avg_dir = Vector2(1, 0)  # fallback

#     avg_dir = avg_dir.normalize()
#     avg_len = (dir1.length() + dir2.length()) * 0.25  # half-length
#     offset = avg_dir * avg_len
#     return (center - offset, center + offset)

class GuidelineBuilder:
    def __init__(self, initial_pos: Vector2, initial_ori: float, car_length: float = 10.0, level_scale: float = 1000.0):
        self.car_forward = Vector2(math.cos(initial_ori), math.sin(initial_ori))
        self.current_line: tuple[Vector2, Vector2] | None = None
        self.previous_line: tuple[Vector2, Vector2] | None = None
        self.current_direction = Vector2(1, 0)
        self.initial_pos = initial_pos
        self.valid_middlelines = []

        self.car_length = car_length
        self.level_scale = level_scale
        self._update_thresholds()

    def _update_thresholds(self):
        self.DIST_THRESHOLD = max(self.car_length * 2.0, self.level_scale * 0.05)
        self.MAX_JUMP = self.level_scale * 0.1

    def build(self, middlelines: list[tuple[Vector2, Vector2]]):
        if not middlelines:
            return

        self.valid_middlelines = self._filter_continuous_segments(middlelines)
        if not self.valid_middlelines:
            return

        closest_line = min(self.valid_middlelines, key=lambda line: self._distance_to_line(self.initial_pos, line))
        self.current_line = self._align_first_line(closest_line)
        self.current_direction = (self.current_line[1] - self.current_line[0]).normalize()

    def _filter_continuous_segments(self, middlelines: list[tuple[Vector2, Vector2]]) -> list[tuple[Vector2, Vector2]]:
        if not middlelines:
            return []

        remaining = middlelines.copy()
        start_line = min(remaining, key=lambda line: self._distance_to_line(self.initial_pos, line))
        base_line = self._align_first_line(start_line)
        path = [base_line]
        remaining.remove(start_line)

        current_line = base_line
        current_dir = (current_line[1] - current_line[0]).normalize()

        while remaining:
            best_score = float('inf')
            best_candidate = None

            for raw in remaining:
                candidate = self._choose_best_orientation_simple(raw, current_dir)
                if candidate is None:
                    continue

                line, direction = candidate
                dist = (current_line[1] - line[0]).length()

                if dist > self.MAX_JUMP:
                    continue

                angle_diff = abs(direction.angle_to(current_dir))
                score = dist + angle_diff * 50

                if score < best_score:
                    best_score = score
                    best_candidate = (raw, line, direction)

            if best_candidate is None:
                break

            raw_line, best_line, best_dir = best_candidate
            path.append(best_line)
            remaining.remove(raw_line)
            current_line = best_line
            current_dir = best_dir

        return path

    def _distance_to_line(self, point: Vector2, line: tuple[Vector2, Vector2]) -> float:
        p1, p2 = line
        if p1 == p2:
            return (point - p1).length()

        line_vec = p2 - p1
        t = max(0, min(1, (point - p1).dot(line_vec) / line_vec.length_squared()))
        projection = p1 + t * line_vec
        return (point - projection).length()

    def _align_first_line(self, raw_line: tuple[Vector2, Vector2]) -> tuple[Vector2, Vector2]:
        p1, p2 = raw_line
        dir = (p2 - p1).normalize()
        return (p1, p2) if dir.dot(self.car_forward) >= 0 else (p2, p1)

    def direction_at(self, position: Vector2) -> Vector2:
        if not self.valid_middlelines:
            return self.car_forward

        if self.current_line:
            if self._distance_to_line(position, self.current_line) < self.DIST_THRESHOLD:
                return self.current_direction

        closest_line = min(self.valid_middlelines, key=lambda line: self._distance_to_line(position, line))
        closest_distance = self._distance_to_line(position, closest_line)
        current_distance = self._distance_to_line(position, self.current_line) if self.current_line else float('inf')

        HYSTERESIS = 0.8
        if self.current_line is None or closest_distance < current_distance * HYSTERESIS:
            self.previous_line = self.current_line
            ref_dir = self.current_direction if self.current_line else self.car_forward
            oriented = self._choose_best_orientation_simple(closest_line, ref_dir)
            if oriented:
                self.current_line, self.current_direction = oriented

        return self.current_direction

    def _choose_best_orientation_simple(self, raw_line: tuple[Vector2, Vector2], ref_dir: Vector2) -> tuple[tuple[Vector2, Vector2], Vector2] | None:
        p1, p2 = raw_line
        dir1 = p2 - p1
        if dir1.length() == 0:
            return None
        dir1 = dir1.normalize()
        dir2 = -dir1

        if ref_dir.dot(dir1) >= ref_dir.dot(dir2):
            return ((p1, p2), dir1)
        else:
            return ((p2, p1), dir2)
