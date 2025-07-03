import pygame
import numpy as np
import math
from quad_func import quad_tree_from_list

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


class Level:
    def __init__(self, walls,checkpoints,BACKGROUND_COLOR = ([0,0,0]), prop=(1000,1000),FPS=60,g=9.81,location=(100,100,0.5*math.pi)):
        # walls should be a list of Line objects
        self.walls = walls
        self.checkpoints=checkpoints
        self.BACKGROUND_COLOR=BACKGROUND_COLOR
        self.proportions=prop
        self.FPS=FPS
        self.g=g
        self.location=location

        self.boolean_walls=quad_tree_from_list(self.walls,self)

    def add_wall(self, x1, y1, x2, y2):
        # Adds a wall (line) to the track
        self.walls.append(Line(x1, y1, x2, y2))
    def reset_quad_tree(self):
        self.boolean_walls=quad_tree_from_list(self.walls,self)
    def draw(self, surface, checkers_truth=True):
        surface.fill(self.BACKGROUND_COLOR)
        # Draw all walls of the track
        for wall in self.walls:
            wall.draw(surface,([0,255,255]))

        # if checkers_truth:
        #     for checkpoint in self.checkpoints:
        #         checkpoint.draw(surface,([0,255,0]))
            
    def checkall(self):
        return all(not cek.turned for cek in self.checkpoints)

class Car:
    # Constructor method to initialize attributesFlevel
    def __init__(self, mass, length, width, color,pull,ni=5,location=(100,100,0.5*math.pi),k=5,turning_speed=0.05):
        self.mass=mass
        self.length=length
        self.width=width
        self.pull=pull
        self.ori=location[2]
        self.ni=ni
        self.bni=ni
        self.k=k
        self.ax=0
        self.ay=0
        
        self.x=location[0]
        self.y=location[1]
        
        
        self.vx=0
        self.vy=0
        self.rfx=0
        self.rfy=0
        self.turning_speed=turning_speed
        
        self.color=color
    def tostart(self,location):
        self.ori=location[2]

        self.ax=0
        self.ay=0
        
        self.x=location[0]
        self.y=location[1]
        
        self.vx=0
        self.vy=0
        self.rfx=0
        self.rfy=0
    def Ori(self):
        self.ori = (self.ori + 2 * math.pi) % (2 * math.pi)
    # Method to display information
    # def show(self, screen):
    #     pygame.draw.rect(screen, self.color, [self.x, self.y,self.width,self.length])
    def decide_quad(self,level):
        quads=[False,False,False,False]
        cos_ori = math.cos(self.ori)
        sin_ori = math.sin(self.ori)
        half_width = self.length / 2
        half_length = self.width / 2
        corners = [
            (-half_width, -half_length),  # Top-left
            (half_width, -half_length),   # Top-right
            (half_width, half_length),    # Bottom-right
            (-half_width, half_length)    # Bottom-left
        ]
        rotated_corners = [
            (
                self.x + x * cos_ori - y * sin_ori,
                self.y + x * sin_ori + y * cos_ori
            )
            for x, y in corners
        ]

        for corner in rotated_corners:
            x, y = corner
            width, height = level.proportions  # Pretpostavljam da proportions = (width, height)

            if x < width / 2 and y < height / 2:
                quads[0]=True
            elif x >= width / 2 and y < height / 2:
                quads[1]=True
            elif x < width / 2 and y >= height / 2:
                quads[2]=True
            elif x >= width / 2 and y >= height / 2:
                quads[3]=True
        return quads

    def show(self, screen):
        cos_ori = math.cos(self.ori)
        sin_ori = math.sin(self.ori)
        half_width = self.length / 2
        half_length = self.width / 2
        corners = [
            (-half_width, -half_length),  # Top-left
            (half_width, -half_length),   # Top-right
            (half_width, half_length),    # Bottom-right
            (-half_width, half_length)    # Bottom-left
        ]
        rotated_corners = [
            (
                self.x + x * cos_ori - y * sin_ori,
                self.y + x * sin_ori + y * cos_ori
            )
            for x, y in corners
        ]
        pygame.draw.polygon(screen, self.color, rotated_corners)
            # Highlight the front of the car
        front_center_x = self.x + half_length * cos_ori
        front_center_y = self.y + half_length * sin_ori

        # Draw a small triangle at the front to indicate the car's direction
        front_triangle = [
            (
                front_center_x + half_width * 0.5 * sin_ori,
                front_center_y - half_width * 0.5 * cos_ori
            ),
            (
                front_center_x - half_width * 0.5 * sin_ori,
                front_center_y + half_width * 0.5 * cos_ori
            ),
            (
                front_center_x + half_length * 0.5 * cos_ori,
                front_center_y + half_length * 0.5 * sin_ori
            )
        ]
        pygame.draw.polygon(screen, (255, 0, 0), front_triangle)  # Red triangle for the front indicator
        



    def ac(self,FPS):
        self.ax=(self.rfx/self.mass)/FPS
        self.vx+=self.ax
        self.ay=(self.rfy/self.mass)/FPS
        self.vy+=self.ay
    def step(self,FPS):
        self.x+=self.vx/FPS
        self.y+=self.vy/FPS
    def gas(self):
        self.rfx+=self.pull*math.cos(self.ori)
        self.rfy+=self.pull*math.sin(self.ori)
    def friction(self,g):

        velocity_magnitude = math.hypot(self.vx, self.vy)

        if(velocity_magnitude!=0):
            self.rfx-=(self.mass*g)*self.ni*(self.vx/velocity_magnitude)
            self.rfy-=(self.mass*g)*self.ni*(self.vy/velocity_magnitude)
    def brake(self):
        self.ni=self.k*self.bni
    def steerleft(self):
        self.ori-=self.turning_speed
    def steerright(self):
        self.ori+=self.turning_speed
        

    def wallinter(self, lines):
        """Check if any line of the car intersects with given lines."""
        cos_ori = math.cos(self.ori)
        sin_ori = math.sin(self.ori)
        half_width = self.length / 2
        half_length = self.width / 2

        # Calculate the four corners of the car
        corners = [
            (-half_width, -half_length),  # Top-left
            (half_width, -half_length),   # Top-right
            (half_width, half_length),    # Bottom-right
            (-half_width, half_length)    # Bottom-left
        ]
        transformed_corners = [
            (
                self.x + x * cos_ori - y * sin_ori,
                self.y + x * sin_ori + y * cos_ori
            )
            for x, y in corners
        ]

        # Create car edges (lines connecting the corners)
        carlines = [
            Line(transformed_corners[i][0], transformed_corners[i][1],
                transformed_corners[(i + 1) % 4][0], transformed_corners[(i + 1) % 4][1])
            for i in range(4)
        ]

        # Check for intersections between car edges and the given lines
        for carline in carlines:
            for line in lines:
                if carline.is_intersecting(line):
                    return True

        return False
    
    def checkinter(self, lines):
        """Check if any line of the car intersects with given lines."""
        cos_ori = math.cos(self.ori)
        sin_ori = math.sin(self.ori)
        half_width = self.length / 2
        half_length = self.width / 2

        # Calculate the four corners of the car
        corners = [
            (-half_width, -half_length),  # Top-left
            (half_width, -half_length),   # Top-right
            (half_width, half_length),    # Bottom-right
            (-half_width, half_length)    # Bottom-left
        ]
        transformed_corners = [
            (
                self.x + x * cos_ori - y * sin_ori,
                self.y + x * sin_ori + y * cos_ori
            )
            for x, y in corners
        ]

        # Create car edges (lines connecting the corners)
        carlines = [
            Line(transformed_corners[i][0], transformed_corners[i][1],
                transformed_corners[(i + 1) % 4][0], transformed_corners[(i + 1) % 4][1])
            for i in range(4)
        ]

        # Check for intersections between car edges and the given lines
        for carline in carlines:
            for line in lines:
                if carline.is_intersecting(line):
                    if(line.turned):
                        # line.turned=False
                        return True

        return False



def calculate_ray_hits(ray_lines, wall_lines):
    intersections = []
    
    for ray in ray_lines:
        closest_point = None
        min_distance_sq = float('inf')
        ray_start = (ray.x1, ray.y1)
        
        for wall in wall_lines:
            if ray.is_intersecting(wall):
                intersection = ray.intersection_point(wall)
                if intersection:
                    # Calculate squared distance from ray's starting point
                    distance_sq = (intersection[0] - ray_start[0])**2 + (intersection[1] - ray_start[1])**2
                    # Update the closest point if necessary
                    if distance_sq < min_distance_sq:
                        min_distance_sq = distance_sq
                        closest_point = intersection
        
        intersections.append(closest_point)
    
    return intersections


def send_intersections(max1, max2, x, y, walls, ori, ray_number):
    rays = []
    ray_length = max(max1, max2) * 3  # Increased length for safety
    angle_increment = 2 * np.pi / (2 * ray_number - 2)

    for i in range(ray_number):
        angle = ori - math.pi / 2 + i * angle_increment
        x2 = x + ray_length * math.cos(angle)
        y2 = y + ray_length * math.sin(angle)
        rays.append(Line(x, y, x2, y2))

    return calculate_ray_hits(rays, walls)

def theVdis(max1, max2, x, y, walls, ori):
    rays = []
    ray_length = max(max1, max2) * 3  # Cast far enough
    x2 = x + ray_length * math.cos(ori)
    y2 = y + ray_length * math.sin(ori)
    rays.append(Line(x, y, x2, y2))
    return calculate_ray_hits(rays, walls)

def draw_intersections(intersections, screen):
    for intersection in intersections:
        if intersection is not None:
            pygame.draw.circle(screen, (255, 150, 0), (int(intersection[0]), int(intersection[1])), 5)



def new_ray_intersection(max1, max2, x, y, ori, ray_number, walls, screen=None,car=None):
    rays = []
    ray_length = max(max1, max2) * 3  # Safe ray length
    angle_increment = 2 * 3.141592653589793 / (2 * ray_number - 2)

    for i in range(ray_number):
        angle = ori - 3.141592653589793 / 2 + i * angle_increment
        x2 = x + ray_length * math.cos(angle)
        y2 = y + ray_length * math.sin(angle)
        rays.append((angle, Line(x, y, x2, y2)))

    mins = []

    for angle, ray in rays:
        min_distance_sq = ray_length * ray_length
        closest_distance = ray_length
        for wall in walls:
            if ray.is_intersecting(wall):
                intersection = ray.intersection_point(wall)
                if intersection:
                    dx = intersection[0] - x
                    dy = intersection[1] - y
                    distance_sq = dx * dx + dy * dy
                    if distance_sq < min_distance_sq:
                        min_distance_sq = distance_sq
                        closest_distance = distance_sq ** 0.5
        mins.append(closest_distance)

    if screen:
        points = []

        # Calculate endpoints of rays
        for i, (angle, _) in enumerate(rays):
            distance = mins[i]
            end_x = x + distance * math.cos(angle)
            end_y = y + distance * math.sin(angle)
            points.append((end_x, end_y))

        # Draw filled polygon connecting all points
        if len(points) >= 3:  # Minimum 3 points required to form a polygon
            pygame.draw.polygon(screen, 'yellow', points, width=0)  # width=0 means filled
        car.show(screen)



    return mins


def vis_single_state(ray_number,state,screen=None):
    rays = []
    ray_length = 4000 # Safe ray length
    angle_increment = 2 * 3.141592653589793 / (2 * ray_number - 2)

    x=0
    y=0


    for i in range(ray_number):
        angle = 0- 3.141592653589793 / 2 + i * angle_increment
        x2 = x + ray_length * math.cos(angle)
        y2 = y + ray_length * math.sin(angle)
        rays.append((angle, Line(x, y, x2, y2)))

    mins = state[-14:]

    if screen:
        points = []

        # Calculate endpoints of rays
        for i, (angle, _) in enumerate(rays):
            distance = mins[i]
            end_x = x + distance * math.cos(angle)
            end_y = y + distance * math.sin(angle)
            points.append((end_x, end_y))

        # Draw filled polygon connecting all points
        if len(points) >= 3:  # Minimum 3 points required to form a polygon
            pygame.draw.polygon(screen, 'yellow', points, width=0)  # width=0 means filled
        
        car=Car(5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)
        car.show(screen)

import cv2
import numpy as np
import math

def vis_single_state_cv2(ray_number, state):
    raysori = []
    raysv=[]
    ray_length = 4000  # Safe ray length
    angle_increment = 2 * math.pi / (2 * ray_number - 2)
    car=Car(5, 40, 20, [100, 200, 255], 1500, 10, (800, 800, 0), 5)

    x = 800  # Center of image
    y = 800

    # Create blank white image
    img_size = 1600
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255





    # Calculate rays
    for i in range(ray_number):
        angle = car.ori-math.pi / 2 + i * angle_increment
        x2 = x + ray_length * math.cos(angle)
        y2 = y + ray_length * math.sin(angle)
        raysori.append((angle, (int(x2), int(y2))))
    for i in range(ray_number):
        angle = -math.pi / 2 + i * angle_increment
        x2 = x + ray_length * math.cos(angle)
        y2 = y + ray_length * math.sin(angle)
        raysv.append((angle, (int(x2), int(y2))))

    mins = state[-ray_number*2:]
    # max(0, min(1, (self.intersections[i] / 2000 + 1) / 2))
    for i in range(ray_number*2):
        mins[i]=(2*mins[i]-1)*2000
    # Calculate points based on distances
    points = []
    for i, (angle, _) in enumerate(rays):
        distance = mins[i]
        print(mins[i])
        end_x = x + distance * math.cos(angle)
        end_y = y + distance * math.sin(angle)
        points.append((int(end_x), int(end_y)))

    # Draw polygon
    if len(points) >= 3:
        cv2.polylines(img, [np.array(points)], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.fillPoly(img, [np.array(points)], color=(0, 255, 255))



    cos_ori = math.cos(car.ori)
    sin_ori = math.sin(car.ori)
    half_width = car.length / 2
    half_length = car.width / 2

    # Define corners relative to center
    corners = [
        (-half_width, -half_length),  # Top-left
        (half_width, -half_length),   # Top-right
        (half_width, half_length),    # Bottom-right
        (-half_width, half_length)    # Bottom-left
    ]

    # Rotate and translate corners
    rotated_corners = [
        (
            int(car.x + x * cos_ori - y * sin_ori),
            int(car.y + x * sin_ori + y * cos_ori)
        )
        for x, y in corners
    ]

    # Draw main body of the car
    cv2.polylines(img, [np.array(rotated_corners)], isClosed=True, color=car.color, thickness=2)
    cv2.fillPoly(img, [np.array(rotated_corners)], color=car.color)

    # Compute front-center of the car
    front_center_x = car.x + half_length * cos_ori
    front_center_y = car.y + half_length * sin_ori

    # Draw red triangle at the front
    front_triangle = [
        (
            int(front_center_x + half_width * 0.5 * sin_ori),
            int(front_center_y - half_width * 0.5 * cos_ori)
        ),
        (
            int(front_center_x - half_width * 0.5 * sin_ori),
            int(front_center_y + half_width * 0.5 * cos_ori)
        ),
        (
            int(front_center_x + half_length * 0.5 * cos_ori),
            int(front_center_y + half_length * 0.5 * sin_ori)
        )
    ]
    cv2.fillPoly(img, [np.array(front_triangle)], color=(0, 0, 255))  # Red triangle
    # # Draw car as a circle
    # car_radius = 10
    # cv2.circle(img, (x, y), car_radius, (255, 0, 0), thickness=-1)

    # Show image
    cv2.imshow("State Visualization", img)
    



def one_ray_intersection(max1, max2, x, y, ori, walls, screen=None):
    ray_length = max(max1, max2) * 3  # Safe long ray
    x2 = x + ray_length * math.cos(ori)
    y2 = y + ray_length * math.sin(ori)
    ray = Line(x, y, x2, y2)

    if screen is not None:
        ray.draw(screen, 'white')

    min_distance_sq = ray_length * ray_length
    closest_distance = ray_length

    for wall in walls:
        if ray.is_intersecting(wall):
            intersection = ray.intersection_point(wall)
            if intersection:
                dx = intersection[0] - x
                dy = intersection[1] - y
                distance_sq = dx * dx + dy * dy
                if distance_sq < min_distance_sq:
                    min_distance_sq = distance_sq
                    closest_distance = distance_sq ** 0.5

    end_x = x + closest_distance * math.cos(ori)
    end_y = y + closest_distance * math.sin(ori)
    final_ray = Line(x, y, end_x, end_y)

    if screen is not None:
        final_ray.draw(screen, 'gray')
        pygame.draw.circle(screen, 'red', (int(end_x), int(end_y)), 5)

    return closest_distance
