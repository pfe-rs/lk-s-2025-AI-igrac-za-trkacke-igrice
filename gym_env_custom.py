import gym
from gym import spaces
import numpy as np
from ClassesML2 import *
from Functions import *
import pygame
import numpy as np
import math
import pygame.surfarray as surfarray
import copy
from agent.const import *
from quad_func import *
from collections import deque

class CustomEnvGA(gym.Env):
    def __init__(self, n_i, level_loc, paramethers,ray_number=7):
        super(CustomEnvGA, self).__init__()

    #     # Action space: 4 discrete actions (gas, brake, left, right)
    #     self.action_space = spaces.MultiBinary(4)

    #     # Observation space: vector of length n_i (e.g., distances or sensors)
    #     self.observation_space = spaces.Box(low=0, high=1, shape=(n_i,), dtype=np.float32)

    #     self.level = level_loader(level_loc)
    #     self.level_copy = copy.deepcopy(self.level)
    #     self.FPS = self.level.FPS
    #     self.car = car_from_parameters(paramethers)
    #     self.car.x=self.level.location[0]
    #     self.car.y=self.level.location[1]

    #     self.ray_number=ray_number

    #     self.state = None
    #     self.score = 0
    #     self.steps = 0
    #     self.run = True

    #     self.check_number=0

        

    # def start_pygame(self):
    #     pygame.init()
    #     pygame.display.set_caption("Zapis")
    #     self.screen = pygame.display.set_mode((self.level.proportions[0], self.level.proportions[1]))
    #     self.clock = pygame.time.Clock()
    #     self.font = pygame.font.Font(None, 36)

    # def reset(self):
    #     self.score = 0
    #     # self.steps = 0
    #     self.check_number=0
    #     self.run = True
    #     self.level = copy.deepcopy(self.level_copy)
    #     self.car.tostart(self.level.location)

    #     # Calculate initial state (e.g., from rays)
    #     self.state = np.array(
    #         new_ray_intersection(self.car.length, self.car.width, self.car.x, self.car.y, self.car.ori,
    #                              self.car.ni, self.level.walls),
    #         dtype=np.float32
    #     )

    #     return self.state

    # def step(self, action):
    #     self.car.rfx = 0
    #     self.car.rfy = 0
    #     self.car.ni = self.car.bni

        

    #     # Process binary actions
    #     if action[0]: self.car.gas()
    #     if action[1]: self.car.brake()
    #     if action[2]: self.car.steerleft()
    #     if action[3]: self.car.steerright()

    #     # Physics and state update
    #     self.car.Ori()
    #     self.car.friction(self.level.g)
    #     self.car.ac(self.FPS)
    #     self.car.step(self.FPS)

    #     # os.system('clear')




    #     # decided_quad=self.car.decide_quad(self.level)
       
    #     # self.chosen_walls= get_chosen_ones(self.level.walls,self.level.boolean_walls,decided_quad)

    #     # print(len(self.chosen_walls))




    #     self.state=[]

    #     # Parametri
    #     self.state.append(max(0, min(1,self.car.mass/100)))

    #     self.state.append(max(0, min(1,self.car.length/200)))
    #     self.state.append(max(0, min(1,self.car.width/100)))

    #     self.state.append(max(0, min(1,self.car.ni/200)))
    #     self.state.append(max(0, min(1,self.car.k/120)))

    #     self.state.append(max(0, min(1,self.car.pull/10000)))

        

    #     # Stanja
    #     self.state.append(max(0, min(1,(math.sin(self.car.ori)+1)/2)))
    #     self.state.append(max(0, min(1,(math.cos(self.car.ori)+1)/2)))

    #     # print(self.car.ni)

    #     self.state.append(max(0, min(1, (self.car.vx / 1000 + 1) / 2)))
    #     self.state.append(max(0, min(1, (self.car.vx / 1000 + 1) / 2)))






    #     self.intersections=[]
    #     self.intersections.extend(new_ray_intersection(self.level.proportions[0],
    #                                     self.level.proportions[1], 
    #                                     self.car.x, 
    #                                     self.car.y,
    #                                     self.car.ori,
    #                                     self.car.ni,
    #                                     self.level.walls))
        
    #     self.intersections.extend(new_ray_intersection(self.level.proportions[0],
    #                                     self.level.proportions[1], 
    #                                     self.car.x, 
    #                                     self.car.y,
    #                                     math.atan2(self.car.vy, self.car.vx),
    #                                     self.car.ni,
    #                                     self.level.walls))
    #     # self.state = np.array(
    #     #     new_ray_intersection(self.car.length, self.car.width, self.car.x, self.car.y, self.car.ori,
    #     #                          self.car.ni, self.level.walls),
    #     #     dtype=np.float32
    #     # )

    #     for i in range(len(self.intersections)):
    #         self.intersections[i]=max(0, min(1, (self.intersections[i] / 2000 + 1) / 2))


    #     self.state.extend(self.intersections)
    #     reward = 0  # You can design this better
    #     done = False
    #     if self.car.wallinter(self.level.walls):
    #         reward = -10
    #         done = True
    #         self.run = False
    #     # if self.car.wallinter(self.chosen_walls):
    #     #     reward = -10
    #     #     done = True
    #     #     self.run = False

    #     if self.car.checkinter([self.level.checkpoints[self.check_number]]):
    #         self.check_number+=1
    #         if self.check_number>len(self.level.checkpoints):
    #             self.check_number-=len(self.level.checkpoints)
    #         reward = 5
    #         self.score += 1

    #     # if self.level.checkall():
    #     #     self.level = copy.deepcopy(self.level_copy)

    #     self.steps += 1


    #     # print(self.state)

    #     return self.state, reward, done, self.steps

    # def render(self):
    #     pygame.display.flip()
    #     self.clock.tick(self.FPS)

    #     # self.level.draw(self.screen,chosen_walls=self.chosen_walls)
    #     self.level.draw(self.screen)

    #     self.level.checkpoints[self.check_number].draw(self.screen,(100,0,100))
    #     self.car.show(self.screen)

    #     textscore = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
    #     textsteps = self.font.render("Steps: " + str(self.steps), True, (255, 255, 255))

    #     self.screen.blit(textscore, (50, 50))
    #     self.screen.blit(textsteps, (50, 100))

    #     # print(f"State: {self.state}")

    #     self.last_screen_array = surfarray.array3d(self.screen)

        
        
    #     # new_ray_intersection(self.level.proportions[0],
    #     #                     self.level.proportions[1],
    #     #                     self.car.x,
    #     #                     self.car.y,
    #     #                     self.car.ori,
    #     #                     self.ray_number,
    #     #                     self.level.walls,
    #     #                     screen=self.screen)

    #     # draw_intersections(self.intersections,self.screen)

    # def close(self):
    #     pygame.quit()




class CustomEnvGAWithQuads(gym.Env):
    def __init__(self, n_i, level_loc, paramethers,ray_number=7):
        super(CustomEnvGAWithQuads, self).__init__()

        # Action space: 4 discrete actions (gas, brake, left, right)
        self.action_space = spaces.MultiBinary(4)

        # Observation space: vector of length n_i (e.g., distances or sensors)
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_i,), dtype=np.float32)

        self.level = level_loader(level_loc)
        self.level_copy = copy.deepcopy(self.level)
        self.FPS = self.level.FPS
        self.car = car_from_parameters(paramethers)
        self.car.x=self.level.location[0]
        self.car.y=self.level.location[1]
        self.car.ori=self.level.location[2]

        self.ray_number=ray_number

        self.state = None
        self.score = 0
        self.steps = 0
        self.run = True

        self.check_number=0

        self.from_state=False


        self.init_replay()





        

    def start_pygame(self):
        pygame.init()
        pygame.display.set_caption("Zapis")
        self.screen = pygame.display.set_mode((self.level.proportions[0], self.level.proportions[1]))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

    def reset(self,startvx=0):
        self.score = 0
        self.steps = 0
        self.check_number=0
        self.run = True
        self.level = copy.deepcopy(self.level_copy)
        self.car.tostart(self.level.location)
        

        # Calculate initial state (e.g., from rays)
        # self.get_state()
        self.init_replay()
        self.update_replay()

        
        #BE CAREFUL

        self.car.vx=startvx


        return self.state
    def init_replay(self):
        self.get_state()
        initial = self.state.copy() + [0, 0, 0, 0]  # Correct way to combine
        self.replay = deque([initial.copy() for _ in range(replay_size+last_size)])
        self.replay.append(self.state.copy())  # Latest state without action

    def update_replay(self, action=[0, 0, 0, 0]):
        self.replay.popleft()
        
        # Overwrite last element to include actionreplay_size
        latest = self.replay[-1] + action
        self.replay[-1] = latest

        # Append new state
        self.get_state()
        self.replay.append(self.state.copy())

    def get_replay(self):
        part_size = replay_size // 10  # 10 parts of 12 frames
        input_replay = []

        # Latest clean state
        input_replay.append(self.replay[-1])

        # last_size previous states WITH actions

        g=-1-last_size
        meow=list(self.replay)[g: -1]
        input_replay.extend(meow)

        # First 120 past states
        replay_list = list(self.replay)[:replay_size]

        for i in range(10):
            part = replay_list[i * part_size : (i + 1) * part_size]
            mean_state = np.mean(part, axis=0)
            input_replay.append(mean_state)

        # os.system('clear')

        # for idx, state in enumerate(input_replay):
        #     print(f"Part {idx}: {state}")

        # print(f"Total parts: {len(input_replay)}")

        # Stack all arrays and flatten to 1D
        input_replay = np.concatenate(input_replay).flatten()

        # print(f"Final flattened length: {len(input_replay)}")
        # self.state
        return input_replay

            

    def get_state(self):
        self.state=[]

        # Parametri
        self.state.append(max(0, min(1,self.car.mass/max_car_mass)))

        self.state.append(max(0, min(1,self.car.length/max_car_length)))
        self.state.append(max(0, min(1,self.car.width/max_car_width)))

        self.state.append(max(0, min(1,self.car.ni/max_car_ni)))
        self.state.append(max(0, min(1,self.car.k/max_car_k)))

        self.state.append(max(0, min(1,self.car.pull/max_car_pull)))

        

        # Stanja
        self.state.append(max(0, min(1,(math.sin(self.car.ori)+1)/2)))
        self.state.append(max(0, min(1,(math.cos(self.car.ori)+1)/2)))

        # print(self.car.ni)

        self.state.append(max(0, min(1, (self.car.vx / max_car_vx + 1) / 2)))
        self.state.append(max(0, min(1, (self.car.vy / max_car_vy + 1) / 2)))






        self.intersections=[]
        if self.from_state:
            screen=self.screen
            car=self.car
        else:
            screen=None
            car=None
        self.intersections.extend(new_ray_intersection(self.level.proportions[0],
                                        self.level.proportions[1], 
                                        self.car.x, 
                                        self.car.y,
                                        self.car.ori,
                                        self.ray_number,
                                        self.level.walls,
                                        screen,
                                        car))
        
        # ntersection(max1, max2, x, y, ori, ray_number, walls, screen=None):
        
        self.intersections.extend(new_ray_intersection(self.level.proportions[0],
                                        self.level.proportions[1], 
                                        self.car.x, 
                                        self.car.y,
                                        math.atan2(self.car.vy, self.car.vx),
                                        self.ray_number,
                                        self.level.walls,
                                        screen,
                                        car))
        
        for i in range(len(self.intersections)):
            self.intersections[i]=max(0, min(1, (self.intersections[i] / 2000 + 1) / 2))


        self.state.extend(self.intersections)
    def step(self, action,from_state=False):
        self.from_state=from_state
        self.car.rfx = 0
        self.car.rfy = 0
        self.car.ni = self.car.bni

        # Process binary actions
        if action[0]: self.car.gas()
        if action[1]: self.car.brake()
        if action[2]: self.car.steerleft()
        if action[3]: self.car.steerright()

        # Physics and state update
        self.car.Ori()
        self.car.friction(self.level.g)
        self.car.ac(self.FPS)
        self.car.step(self.FPS)

        # os.system('clear')




        decided_quad=self.car.decide_quad(self.level)
        self.chosen_walls= get_chosen_ones(self.level.walls,self.level.boolean_walls,decided_quad)

        # print(len(self.chosen_walls))




        self.update_replay(action)
        # for frame in self.replay:
        #     print(frame[8:10])
        # print(self.replay,end="\n")
        # self.state = np.array(
        #     new_ray_intersection(self.car.length, self.car.width, self.car.x, self.car.y, self.car.ori,
        #                          self.car.ni, self.level.walls),
        #     dtype=np.float32
        # )

        reward = 0  # You can design this better
        done = False
        # if self.car.wallinter(self.level.walls):
        #     reward = -10
        #     done = True
        #     self.run = False
        if self.car.wallinter(self.chosen_walls):
            # reward = -10
            done = True
            self.run = False

        if self.car.checkinter([self.level.checkpoints[self.check_number]]):
            self.check_number+=1
            if self.check_number>=len(self.level.checkpoints):
                self.check_number-=len(self.level.checkpoints)
            reward = 5
            self.score += 1

        self.steps += 1

        return self.state, reward, done, self.steps,self.score

    def render(self):
        pygame.display.flip()
        self.clock.tick(self.FPS)

        if not self.from_state:
            self.level.draw(self.screen,chosen_walls=self.chosen_walls)
            # self.level.draw(self.screen)

            self.level.checkpoints[self.check_number].draw(self.screen,(100,0,100))
            self.car.show(self.screen)

            textscore = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
            textsteps = self.font.render("Steps: " + str(self.steps), True, (255, 255, 255))

            self.screen.blit(textscore, (50, 50))
            self.screen.blit(textsteps, (50, 100))

            self.last_screen_array = surfarray.array3d(self.screen)
        else:
            self.screen.fill(self.level.BACKGROUND_COLOR)
            self.last_screen_array = surfarray.array3d(self.screen)

            textscore = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
            textsteps = self.font.render("Steps: " + str(self.steps), True, (255, 255, 255))

            self.screen.blit(textscore, (50, 50))
            self.screen.blit(textsteps, (50, 100))

    
    def close(self):
        pygame.quit()








class CustomEnvGAWithQuadsMath(gym.Env):
    def __init__(self, level_loc, paramethers):
        super(CustomEnvGAWithQuadsMath, self).__init__()

        self.level = level_loader(level_loc)
        self.level_copy = copy.deepcopy(self.level)
        self.FPS = self.level.FPS
        self.car = car_from_parameters(paramethers)
        self.car.x=self.level.location[0]
        self.car.y=self.level.location[1]
        self.car.ori=self.level.location[2]

        self.score = 0
        self.steps = 0
        self.run = True

        self.check_number=0

        self.from_state=False




        

    def start_pygame(self):
        pygame.init()
        pygame.display.set_caption("Zapis")
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

    def reset(self):
        self.score = 0
        self.steps = 0
        self.check_number=0
        self.run = True
        self.level = copy.deepcopy(self.level_copy)
        self.car.tostart(self.level.location)
        return 
    
    def step(self, action,from_state=False):
        self.from_state=from_state
        self.car.rfx = 0
        self.car.rfy = 0
        self.car.ni = self.car.bni

        # Process binary actions
        if action[0]: self.car.gas()
        if action[1]: self.car.brake()
        if action[2]: self.car.steerleft()
        if action[3]: self.car.steerright()

        # Physics and state update
        self.car.Ori()
        self.car.friction(self.level.g)
        self.car.ac(self.FPS)
        self.car.step(self.FPS)

        # os.system('clear')
        decided_quad=self.car.decide_quad(self.level)
        self.chosen_walls= get_chosen_ones(self.level.walls,self.level.boolean_walls,decided_quad)

        reward = 0  # You can design this better
        done = False

        if self.car.wallinter(self.chosen_walls):
            # reward = -10
            done = True
            self.run = False
        if self.car.checkinter([self.level.checkpoints[self.check_number]]):
            self.check_number+=1
            if self.check_number>=len(self.level.checkpoints):
                self.check_number-=len(self.level.checkpoints)
            reward = 5
            self.score += 1

        self.steps += 1

        return reward, done, self.steps

    def render(self):
        pygame.display.flip()
        self.clock.tick(self.FPS)

        if not self.from_state:
            self.level.draw(self.screen)
            # self.level.draw(self.screen)

            self.level.checkpoints[self.check_number].draw(self.screen,(100,0,100))
            self.car.show(self.screen)

            textscore = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
            textsteps = self.font.render("Steps: " + str(self.steps), True, (255, 255, 255))

            self.screen.blit(textscore, (50, 50))
            self.screen.blit(textsteps, (50, 100))

            self.last_screen_array = surfarray.array3d(self.screen)
        else:
            self.screen.fill(self.level.BACKGROUND_COLOR)
            self.last_screen_array = surfarray.array3d(self.screen)

            textscore = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
            textsteps = self.font.render("Steps: " + str(self.steps), True, (255, 255, 255))

            self.screen.blit(textscore, (50, 50))
            self.screen.blit(textsteps, (50, 100))

    
    def close(self):
        pygame.quit()
