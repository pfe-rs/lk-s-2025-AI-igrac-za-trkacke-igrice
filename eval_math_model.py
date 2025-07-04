import sys
from agent.utils import get_device
# from perfect_math_drift_master import MathModel
from perfect_math import MathModel
from gym_env_custom import CustomEnvGAWithQuadsMath
import time
import torch
import random
import os
import math
import numpy as np
# python clean-codes/single_model_runner.py models_supervised/model1.pkl levels/10.pkl

def env_fn():
    return CustomEnvGAWithQuadsMath(level_file, car_params)

ray_number = 7
parametri = 6
stanja = 4
n_inputs = parametri + stanja + 2 * ray_number
n_inputs=584
maxsteps=1500


os.system('clear')

rewards=[]

seconds=30
maxsteps=seconds*60


track_width_pixels=100
track_width_meters=12
pixel_per_meter=track_width_pixels/track_width_meters

# car_params = (1200, int(3.4*pixel_per_meter), int(1.7*pixel_per_meter), [100, 200, 255], 20000*pixel_per_meter, 0.05, (0, 0, 0), 9,0.083)
# car_params = (5, 40, 20, [100, 200, 255], 20000, 10, (0, 0, 0), 400)
car_params = (5, 40, 20, [100, 200, 255], 1000, 10, (0, 0, 0), 40)
# car_params = (5, 40, 20, [100, 200, 255], 45000, 10, (0, 0, 0), 150)



model = MathModel(0.3,1,side_offset = 20,pixel_per_meter=pixel_per_meter)
# model = MathModel(-0.3,0,side_offset = 40,pixel_per_meter=pixel_per_meter)


# lvl_num=random.randint(1,11)
for lvl_num in range(12):
    print("\n")

    start_time = time.time()

    level_file="clean-codes/levels/"+str(lvl_num)+".pkl"
    print("Math solving the level number "+str(lvl_num)+": ")

    env=env_fn()


    # reward=model.run_in_environment(env,visualize=True ,visualize_gb=True,visualize_lr=True,maxsteps=maxsteps)
    reward=model.run_in_environment(env,visualize=False ,maxsteps=maxsteps)
    # (self, env, visualize=True, maxsteps=500, device="cuda",startvx=0,startstep=0,plotenzi_loc=None)
    # best_model.run_in_environment(env_fn(), visualize=True, threshold=0.5,maxsteps=200)
    real_reward=reward/5/len(env.level.checkpoints)
    print("Loops: "+str(real_reward))
    rewards.append(real_reward)
    end_time = time.time()
    asas=model.average_speed
    print("Speed: "+str(asas)+"km/h")
    print(f"Execution time: {end_time - start_time:.6f} seconds")

print("\n")
print(np.mean(rewards))
