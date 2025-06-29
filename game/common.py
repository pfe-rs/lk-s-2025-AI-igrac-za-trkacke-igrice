from pygame import Vector2
import pygame


class Line:
    def __init__(self, x1, y1, x2, y2):
        # Line defined by two points (x1, y1) and (x2, y2)
        if (x1, y1) == (x2, y2):
            raise ValueError("A line must have two distinct endpoints.")
        
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.turned = True

    def draw(self, surface, color):
        # Draw the line on a given surface (e.g., using Pygame) if turned is True
        if self.turned:
            pygame.draw.line(surface, color, (self.x1, self.y1), (self.x2, self.y2))
    def get_bounds(self):
        """Vraca pravougaonik koji pokriva liniju."""
        x_min = min(self.x1, self.x2)
        y_min = min(self.y1, self.y2)
        x_max = max(self.x1, self.x2)
        y_max = max(self.y1, self.y2)
        return (x_min, y_min, x_max, y_max)
    def is_intersecting(self, other_line):
        """Check if this line intersects with another line."""
        def ccw(A, B, C):
            """Check if the points A, B, C are counterclockwise."""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        # Points of the two lines
        A = (self.x1, self.y1)
        B = (self.x2, self.y2)
        C = (other_line.x1, other_line.y1)
        D = (other_line.x2, other_line.y2)

        # Check if the line segments AB and CD intersect
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def intersection_point(self, other_line):
        """Find the intersection point with another line."""
        def line_intersection(p1, p2, q1, q2):
            """Find intersection point between line segments p1p2 and q1q2."""
            d1 = [p2[0] - p1[0], p2[1] - p1[1]]
            d2 = [q2[0] - q1[0], q2[1] - q1[1]]
            det = d1[0] * d2[1] - d1[1] * d2[0]

            if abs(det) < 1e-10:
                return None  # Lines are parallel or coincident

            diff = [q1[0] - p1[0], q1[1] - p1[1]]
            t = (diff[0] * d2[1] - diff[1] * d2[0]) / det
            u = (diff[0] * d1[1] - diff[1] * d1[0]) / det

            if 0 <= t <= 1 and 0 <= u <= 1:
                return (p1[0] + t * d1[0], p1[1] + t * d1[1])
            return None

        p1 = (self.x1, self.y1)
        p2 = (self.x2, self.y2)
        q1 = (other_line.x1, other_line.y1)
        q2 = (other_line.x2, other_line.y2)

        return line_intersection(p1, p2, q1, q2)
    
def line_to_tuple(wall: Line) -> tuple[Vector2, Vector2]:
        return (Vector2(wall.x1, wall.y1), Vector2(wall.x2, wall.y2))
