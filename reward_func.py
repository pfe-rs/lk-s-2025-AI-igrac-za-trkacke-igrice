from ClassesML2 import *
from Functions import *


def phase1_reward(intersections,collision_with_wall,safety_threshold):
    reward = 0
    
    # Primary objective: Stay alive
    if collision_with_wall:
        reward -= 100  # Strong negative for collision (terminal)
    else:
        reward += 1   # Small positive for surviving each timestep
    
    # Optional: Distance-based shaping
    mmm=min(intersections)
    if  mmm< safety_threshold:
        reward -= (safety_threshold - mmm) * 0.1
    
    return reward
