from ClassesML2 import *
import math
from agent.const import *
import os




class MathModel():
    def __init__(self,kz,side_offset):
        self.angle_step_gb=angle_of_view_gb/(all_rays_gb-1)
        self.angle_step_lr=angle_of_view_lr/(all_rays_lr-1)
        self.ray_values_gb=None
        self.ray_values_lr=None
        self.kz=kz
        self.side_offset = side_offset
        # self.lasts[]

    def reset_rays(self,env):
        self.ray_values_gb=  self.new_ray_intersection(env,angle_of_view_gb,all_rays_gb)
        self.ray_values_lr=  self.new_ray_intersection_updated(env,angle_of_view_lr,all_rays_lr)

    def gb_decide(self,env,visualize=False):
        vx=env.car.vx
        vy=env.car.vy
        g=env.level.g
        kb=env.car.k
        ori = math.atan2(vy, vx)
        start_ori=ori-angle_of_view_gb/2
        brake_lenghts=[]
        for i in range(all_rays_gb):
            angle = start_ori+ i * self.angle_step_gb
            # speed_along_ori = vx * math.cos(angle) + vy * math.sin(angle)
            vxao=vx * math.cos(angle)
            vyao=vy * math.sin(angle)

            sbz=(vxao**2+vyao**2)*(1+self.kz)/(2*g*kb)+max(env.car.length,env.car.width)
            if visualize:
                line=Line(env.car.x,env.car.y,env.car.x+math.cos(angle)*self.ray_values_gb[i],env.car.y+math.sin(angle)*self.ray_values_gb[i])
                line.draw(env.screen,([255,255,255]))

                line=Line(env.car.x,env.car.y,env.car.x+math.cos(angle)*sbz,env.car.y+math.sin(angle)*sbz)
                line.draw(env.screen,([100,0,100]))
            brake_lenghts.append(sbz)

        for i in range(len(brake_lenghts)):
            angle = start_ori+ i * self.angle_step_gb
            sbz=brake_lenghts[i]
            if(sbz>=self.ray_values_gb[i]):
                return [0,1]
            
        return [1,0]
        
    def lr_decide(self, env,visualize=False):
        max_index = self.ray_values_lr.index(min(self.ray_values_lr))  # Ray with most space
        v_ori = math.atan2(env.car.vy, env.car.vx)  # Current car orientation
        start_ori = v_ori - angle_of_view_lr / 2
        angle = start_ori + max_index * self.angle_step_lr  # Direction of the best ray

        # Normalize angles to [-pi, pi]
        def normalize_angle(a):
            while a > math.pi:
                a -= 2 * math.pi
            while a < -math.pi:
                a += 2 * math.pi
            return a

        diff = normalize_angle(angle - v_ori)
        os.system('clear')
        print("angle of the best v: "+str(angle))
        print("angle of the model: "+str(v_ori))

        if diff > 0.05:
            return [1,0] # Steer Left
        elif diff < -0.05:
            return [0, 1]  # Steer Right
        else:
            return [0, 0]  # Stay Straight
        
       

    
    def new_ray_intersection(self,env,angle_of_view,all_rays):
        angle_step=angle_of_view/(all_rays-1)
        max1=env.level.proportions[0]
        max2=env.level.proportions[1]
        x=env.car.x
        y=env.car.y
        ori = math.atan2(env.car.vy, env.car.vx)
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
    
    def new_ray_intersection_updated(self, env,angle_of_view,all_rays):
        max1 = env.level.proportions[0]
        max2 = env.level.proportions[1]
        x = env.car.x
        y = env.car.y
        ori = math.atan2(env.car.vy, env.car.vx)
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

                # line=Line(env.car.x,env.car.y,env.car.x+math.cos(angle)*min_distance,env.car.y+math.sin(angle)*min_distance)
                # line.draw(env.screen,([150,150,0]))
                

            distances.append(min_distance)

            

        return distances

    
    def result(self,env,visualize=False):
        self.reset_rays(env)
        return self.gb_decide(env,visualize)+self.lr_decide(env,visualize)

    def run_in_environment(self, env, visualize=True, maxsteps=500):
        
        if visualize:
            env.start_pygame()

        total_reward = 0
        running = True

        while running:
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            running = False

            # Move state to device
            

            # Move actions back to CPU for compatibility with environment
            chosen_actions = self.result(env,visualize=True)

            reward, done, steps = env.step(chosen_actions)
            total_reward += reward

            if visualize:
                env.render()

            if done or steps >= maxsteps:
                running = False
        env.close()
        return total_reward
    

