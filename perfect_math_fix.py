from ClassesML2 import *
import math
from agent.const import *
import os
from tqdm import tqdm
from collections import deque
import copy



class MathModel():
    def __init__(self,kz,kn,side_offset,pixel_per_meter=1):
        self.angle_step_gb=angle_of_view_gb/(all_rays_gb-1)
        self.angle_step_lr=angle_of_view_lr/(all_rays_lr-1)
        self.ray_values_gb=None
        self.ray_values_lr=None
        self.kz=kz
        self.kn=kn
        self.side_offset = side_offset
        self.v_ori=None
        self.v=None
        self.pixel_per_meter=pixel_per_meter
        self.last_ori=None
        # self.lasts[]

    def reset_rays(self,env,visualize=False):
        self.ray_values_gb=  self.new_ray_intersection(env,angle_of_view_gb,all_rays_gb)
        self.ray_values_lr=  self.new_ray_intersection_updated(env,angle_of_view_lr,all_rays_lr,visualize)

    def gb_decide(self,env,visualize=False):
        vx=env.car.vx
        vy=env.car.vy
        g=env.level.g
        kb=env.car.k
        ori = self.v_ori
        start_ori=ori-angle_of_view_gb/2
        brake_lenghts=[]
        friction_lengths=[]
        for i in range(all_rays_gb):
            angle = start_ori+ i * self.angle_step_gb
            # speed_along_ori = vx * math.cos(angle) + vy * math.sin(angle)
            vxao=vx * math.cos(angle)
            vyao=vy * math.sin(angle)

            # sbz=(vxao**2+vyao**2)*(1+self.kz)/(2*g*kb*env.car.ni)+max(env.car.length,env.car.width)
            sbz=(vxao**2+vyao**2)*(1+self.kz)/(2*g*kb*env.car.bni)+max(env.car.length,env.car.width)
            # sfz=
            if visualize:
            #     # line=Line(env.car.x,env.car.y,env.car.x+math.cos(angle)*self.ray_values_gb[i],env.car.y+math.sin(angle)*self.ray_values_gb[i])
            #     # line.draw(env.screen,([255,255,255]))

                line=Line(env.car.x,env.car.y,env.car.x+math.cos(angle)*sbz,env.car.y+math.sin(angle)*sbz)
                line.draw(env.screen,([100,0,100]))
            brake_lenghts.append(sbz)
        
        for i in range(len(brake_lenghts)):
            angle = start_ori+ i * self.angle_step_gb
            sbz=brake_lenghts[i]
            if(self.kn*(sbz)>=self.ray_values_gb[i]):
                return [0,1]
        # print(brake_lenghts)
        for i in range(len(brake_lenghts)):
            # env.car.color=([255,0,0])
            # print("JENASFIBAIBAIOBOIWBFOIABIOWBB")
            angle = start_ori+ i * self.angle_step_gb
            sbz=brake_lenghts[i]

            # line=Line(env.car.x,env.car.y,env.car.x+math.cos(angle)*self.ray_values_gb[i],env.car.y+math.sin(angle)*self.ray_values_gb[i])
            # line.draw(env.screen,([255,255,255]))

            # line=Line(env.car.x,env.car.y,env.car.x+math.cos(angle)*sbz,env.car.y+math.sin(angle)*sbz)
            # line.draw(env.screen,([100,0,100]))
            if(sbz<self.ray_values_gb[i]):
                return [1,0]

        
            
        return [0,0]
        
    def lr_decide(self, env,visualize=False):
        max_index = self.ray_values_lr.index(max(self.ray_values_lr))  # Ray with most space
        v_ori = self.v_ori # Current car orientation
        start_ori = v_ori - angle_of_view_lr /2
        angle = start_ori + max_index * self.angle_step_lr  # Direction of the best ray

        # Normalize angles to [-pi, pi]
        def normalize_angle(a):
            while a > math.pi:
                a -= 2 * math.pi
            while a < -math.pi:
                a += 2 * math.pi
            return a

        diff = normalize_angle(angle - env.car.ori)

        # print("angle of the best v: "+str(angle))
        # print("angle of the model: "+str(env.car.ori))

        if diff > 0.05:
            return  [0, 1] # Steer Left
        elif diff < -0.05:
            return [1, 0]  # Steer Right
        else:
            return [0, 0]  # Stay Straight
        
       

    
    def new_ray_intersection(self,env,angle_of_view,all_rays):
        angle_step=angle_of_view/(all_rays-1)
        max1=env.level.proportions[0]
        max2=env.level.proportions[1]
        x=env.car.x
        y=env.car.y
        ori = self.v_ori
        walls=env.level.walls

        rays = []

        ray_length = max(max1, max2) * 3  # Safe ray length

        start_ori=ori-angle_of_view/2

        
        for i in range(all_rays):
            angle = start_ori+ i * angle_step
            x2 = x + ray_length * math.cos(angle)
            y2 = y + ray_length * math.sin(angle)
            rays.append((angle, Line(x, y, x2, y2)))

        distances = []

        for angle, ray in rays:
            closest_distance = ray_length
            for wall in walls:
                if ray.is_intersecting(wall):
                    intersection = ray.intersection_point(wall)
                    if intersection:
                        dx = intersection[0] - x
                        dy = intersection[1] - y
                        distance = math.hypot(dx, dy)
                        if distance < closest_distance:
                            closest_distance = distance
            distances.append(closest_distance)

        return distances
    
    def new_ray_intersection_updated(self, env,angle_of_view,all_rays,visualize=False):
        max1 = env.level.proportions[0]
        max2 = env.level.proportions[1]
        x = env.car.x
        y = env.car.y
        ori = self.v_ori
        walls = env.level.walls

        rays = []
        ray_length = max(max1, max2) * 3  # Safe ray length

        start_ori = ori - angle_of_view / 2

        # Offset distance for parallel rays on each side

        distances = []

        for i in range(all_rays):
            angle = start_ori + i * self.angle_step_lr

            # Calculate original ray line
            x2 = x + ray_length * math.cos(angle)
            y2 = y + ray_length * math.sin(angle)
            original_ray = Line(x, y, x2, y2)

            # Calculate perpendicular offset vector (to shift ray sideways)
            # Perp vector = (-sin(angle), cos(angle)) normalized * side_offset
            
            offset_dx = -math.sin(angle) * self.side_offset
            offset_dy = math.cos(angle) * self.side_offset

            # Left shifted ray start and end points
            x_left_start = x + offset_dx
            y_left_start = y + offset_dy
            x_left_end = x_left_start + ray_length * math.cos(angle)
            y_left_end = y_left_start + ray_length * math.sin(angle)
            left_ray = Line(x_left_start, y_left_start, x_left_end, y_left_end)

            # Right shifted ray start and end points
            x_right_start = x - offset_dx
            y_right_start = y - offset_dy
            x_right_end = x_right_start + ray_length * math.cos(angle)
            y_right_end = y_right_start + ray_length * math.sin(angle)
            right_ray = Line(x_right_start, y_right_start, x_right_end, y_right_end)

            # Check distances for all three rays, pick minimum
            min_distance = ray_length

            # good_ray=True
            # connector=Line(left_ray.x1,left_ray.y1,right_ray.x1,right_ray.y1)
            # connector.draw(env.screen,(50,250,50))
            # for wall in walls:
            #     if connector.is_intersecting(wall):
            #         good_ray=False
            #         min_distance=0
            #         break

            # if good_ray:
            for ray in [original_ray, left_ray, right_ray]:
            
                for wall in walls:
                    if ray.is_intersecting(wall):
                        intersection = ray.intersection_point(wall)
                        if intersection:
                            dx = intersection[0] - x
                            dy = intersection[1] - y
                            distance = math.hypot(dx, dy)
                            if distance < min_distance:
                                min_distance = distance

            for ray in [original_ray, left_ray, right_ray]:
                # line=Line(env.car.x,env.car.y,env.car.x+math.cos(angle)*min_distance,env.car.y+math.sin(angle)*min_distance)
                if visualize:
                    ray.draw(env.screen,([150,150,0]))
                

            distances.append(min_distance)

            

        return distances

    
    def result(self,env,visualize_gb=False,visualize_lr=False):
        self.v_ori = math.atan2(env.car.vy, env.car.vx)
        
        self.v=math.hypot(env.car.vx,env.car.vy)
        if(abs(self.v)<5):
            self.v_ori=env.car.ori

        # print([self.last_ori,self.v_ori])
        # os.system('clear')
        print("Speed: "+str(self.v/self.pixel_per_meter*3.6)+"km/h")
        self.reset_rays(env,visualize_lr)
        decision=self.gb_decide(env,visualize_gb)+self.lr_decide(env,visualize_lr)
        self.last_ori=self.v_ori
        
        return decision

    def run_in_environment(self, env, visualize=True,visualize_gb=True,visualize_lr=True, maxsteps=500,label=69):
        self.last_ori=env.car.ori
        color_copy=copy.deepcopy(env.car.color)
        

        if visualize:
            env.start_pygame()
        else:
            visualize_gb=False
            visualize_lr=False

        total_reward = 0
        running = True
        for i in tqdm(range(maxsteps), desc="Applying math to "+str(label)+". level"):
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            running = False

            # Move state to device
            
            # os.system('clear')
            

            # Move actions back to CPU for compatibility with environment
            chosen_actions = self.result(env,visualize_gb,visualize_lr)
            print(chosen_actions)

            reward, done, steps = env.step(chosen_actions)
            # print(env.steps)
            total_reward += reward
            # if chosen_actions[0]:
            #     env.car.color=copy.deepcopy(color_copy)
            # else:
            #     env.car.color=([255,0,0])
            if visualize:
                env.render()

            if done or steps >= maxsteps:
                running = False
                break
        env.close()
        return total_reward
    

