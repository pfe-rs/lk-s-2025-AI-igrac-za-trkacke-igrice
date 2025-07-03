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

# lvl_num=random.randint(1,11)
for lvl_num in range(5,6):
    start_time = time.time()
        

    level_file="clean-codes/levels/"+str(lvl_num)+".pkl"

    seconds=60
    maxsteps=seconds*60

    car_params = (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5,0.3)
    # car_params = (5, 40, 20, [100, 200, 255], 3000, 10, (0, 0, 0), 15)
    # car_params = (5, 40, 20, [100, 200, 255], 500, 10, (0, 0, 0), 15)



    model = MathModel(0.5,2,side_offset = 20)
    env=env_fn()
    reward=model.run_in_environment(env, visualize=True,maxsteps=maxsteps,label=lvl_num)
    # (self, env, visualize=True, maxsteps=500, device="cuda",startvx=0,startstep=0,plotenzi_loc=None)
    # best_model.run_in_environment(env_fn(), visualize=True, threshold=0.5,maxsteps=200)
    percent=(reward/5*100)/len(env.level.checkpoints)
    
    rewards.append(percent)
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.6f} seconds")
    print("Reward: "+str(percent)+"%",end="\n\n")

print(np.mean(rewards))
